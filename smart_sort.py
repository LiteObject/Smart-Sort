#!/usr/bin/env python3
"""smart_sort.py - organize a folder by what its files are about.

Pipeline: collect files -> extract text or media -> embed -> cluster -> plan -> move.

Safety model
------------
Nothing is ever moved unless ``--apply`` is passed; the default is a dry run
that prints what would happen. Every applied run appends a JSONL manifest under
``<root>/.smart_sort/`` and ``--undo <manifest> --apply`` replays it in reverse,
verifying SHA-256 checksums first. Destination collisions get a numeric suffix
instead of overwriting anything.

Usage
-----
    python smart_sort.py "D:\\Downloads"                       # preview
    python smart_sort.py "D:\\Downloads" --apply               # reorganize
    python smart_sort.py . --backend tfidf --k 6 --recursive
    python smart_sort.py --undo ".smart_sort/manifest-20250101-120000.jsonl" --apply
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from text_extract import ExtractionError, extract_text, extract_worker, is_heavy

LOG = logging.getLogger("smart-sort")

STATE_DIR = ".smart_sort"
CLUSTER_DIR_RE = re.compile(r"^Cluster_\d+", re.IGNORECASE)

DEFAULT_EXCLUDES: tuple[str, ...] = (
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vs",
    ".vscode",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "dist",
    "build",
    "target",
    "$RECYCLE.BIN",
    "System Volume Information",
)

# Files with these suffixes are media handled by multimodal backends.
MEDIA_SUFFIXES: dict[str, str] = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image",
    ".bmp": "image", ".gif": "image", ".tif": "image", ".tiff": "image",
    ".heic": "image", ".avif": "image",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".m4a": "audio",
    ".ogg": "audio", ".opus": "audio", ".aac": "audio", ".wma": "audio",
    ".mp4": "video", ".mov": "video", ".mkv": "video", ".avi": "video",
    ".webm": "video", ".m4v": "video", ".wmv": "video", ".flv": "video",
}


def media_kind(path: Path) -> str | None:
    return MEDIA_SUFFIXES.get(path.suffix.lower())


# --------------------------------------------------------------------------- #
# Configuration and data types
# --------------------------------------------------------------------------- #


@dataclass
class Settings:
    root: Path
    k: int | None = None
    auto_k: bool = False
    apply: bool = False
    recursive: bool = False
    backend: str = "auto"
    model_path: str | None = None
    n_ctx: int = 2048
    media: tuple[str, ...] = ("image", "audio", "video")
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 32
    text_prompt: str | None = None
    max_chars: int = 4000
    max_chunks: int = 4
    max_pdf_pages: int = 20
    min_chars: int = 20
    seed: int = 0
    label_clusters: bool = True
    use_cache: bool = True
    jobs: int | None = None
    timings: bool = False
    verbose_moves: bool = False
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES


@dataclass
class Document:
    path: Path
    text: str
    chunks: list[str]
    kind: str = "text"


@dataclass
class Skip:
    path: Path
    reason: str


@dataclass
class Move:
    src: Path
    dst: Path
    cluster: int


@dataclass
class ApplyResult:
    moved: int
    failed: int
    manifest: Path | None


class Stopwatch:
    """Collect phase durations for ``--timings`` output."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.phases: dict[str, float] = {}

    def phase(self, name: str) -> "_PhaseTimer":
        return _PhaseTimer(self, name)

    def elapsed_total(self) -> float:
        return time.perf_counter() - self.started

    def print_summary(self) -> None:
        print("\nTimings:")
        for name, seconds in self.phases.items():
            print(f"  {name:<12} {seconds:8.3f} s")
        print(f"  {'wall clock':<12} {self.elapsed_total():8.3f} s")


class _PhaseTimer:
    def __init__(self, stopwatch: Stopwatch, name: str) -> None:
        self._stopwatch = stopwatch
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "_PhaseTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stopwatch.phases[self._name] = time.perf_counter() - self._start


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    folded = name.casefold()
    return any(fnmatch.fnmatch(folded, pattern.casefold()) for pattern in patterns)


def _skip_dir(name: str, excludes: Sequence[str]) -> bool:
    """Skip hidden dirs, previous output dirs, and user excludes."""
    if name.startswith(".") or CLUSTER_DIR_RE.match(name):
        return True
    return _matches_any(name, excludes)


_WINDOWS_HIDDEN_MASK = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2) | getattr(
    stat, "FILE_ATTRIBUTE_SYSTEM", 0x4
)


def is_hidden_or_system(path: Path) -> bool:
    """True for Office lock files and hidden/system entries.

    Windows uses file attributes, so a leading dot is *not* treated as hidden
    there (e.g. ``.NET Consultant notes.docx`` is a real document). On POSIX
    systems the dot prefix convention applies.
    """
    if path.name.startswith("~$"):
        return True
    if os.name != "nt":
        return path.name.startswith(".")
    try:
        attributes = path.stat().st_file_attributes
    except OSError:
        return True
    return bool(attributes & _WINDOWS_HIDDEN_MASK)


def collect_files(
    root: Path,
    recursive: bool = False,
    excludes: Sequence[str] = DEFAULT_EXCLUDES,
) -> list[Path]:
    """Return candidate files, deterministically ordered, symlinks excluded."""
    files: list[Path] = []

    def consider(path: Path) -> None:
        if path.is_symlink() or not path.is_file() or is_hidden_or_system(path):
            return
        if _matches_any(path.name, excludes):
            return
        files.append(path)

    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(
                d
                for d in dirnames
                if not _skip_dir(d, excludes)
                and not is_hidden_or_system(Path(dirpath) / d)
                and not (Path(dirpath) / d).is_symlink()
            )
            for name in sorted(filenames):
                consider(Path(dirpath) / name)
    else:
        for path in sorted(root.iterdir()):
            consider(path)

    return files


def split_chunks(text: str, max_chars: int, max_chunks: int) -> list[str]:
    """Split text into at most ``max_chunks`` windows of ``max_chars`` chars.

    Long documents keep both their opening and closing content so the average
    embedding is not biased toward the first page.
    """
    text = text.strip()
    if not text:
        return []
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text]

    windows = min(max_chunks, math.ceil(len(text) / max_chars))
    if windows <= 1:
        return [text[:max_chars]]
    if len(text) <= max_chars * windows:
        size = math.ceil(len(text) / windows)
        return [text[i * size : (i + 1) * size] for i in range(windows) if text[i * size : (i + 1) * size]]

    last_start = len(text) - max_chars
    starts = {round(i * last_start / (windows - 1)) for i in range(windows)}
    return [text[start : start + max_chars] for start in sorted(starts)]


_PDF_BYTES_PER_SECOND = 30 * 1024 * 1024
_DOCX_BYTES_PER_SECOND = 10 * 1024 * 1024
_PARALLEL_THRESHOLD_SECONDS = 0.75


def _auto_worker_count(heavy: Sequence[Path]) -> int:
    """Parallelize only when expected parse time outweighs process startup.

    Worker startup costs a few hundred milliseconds, so small batches are
    faster sequentially. Throughput constants are deliberately conservative.
    """
    if len(heavy) < 2:
        return 1
    estimated_seconds = 0.0
    try:
        for path in heavy:
            size = path.stat().st_size
            if path.suffix.lower() == ".pdf":
                estimated_seconds += size / _PDF_BYTES_PER_SECOND
            else:
                estimated_seconds += size / _DOCX_BYTES_PER_SECOND
    except OSError:
        return 1
    if estimated_seconds < _PARALLEL_THRESHOLD_SECONDS:
        return 1
    return max(1, min(8, os.cpu_count() or 1, len(heavy)))


def _parallel_heavy_results(
    heavy: Sequence[Path], max_pdf_pages: int, jobs: int
) -> dict[Path, tuple[str | None, str | None]]:
    """Extract PDF/DOCX files in worker processes."""
    workers = max(1, min(jobs, len(heavy)))
    payload = [(str(path), max_pdf_pages) for path in heavy]
    results: dict[Path, tuple[str | None, str | None]] = {}
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as pool:
        for path, outcome in zip(heavy, pool.map(extract_worker, payload)):
            results[path] = outcome
    LOG.info("Extracted %d PDF/DOCX files with %d worker processes", len(heavy), workers)
    return results


def load_documents(
    files: Sequence[Path],
    max_chars: int,
    max_chunks: int,
    max_pdf_pages: int,
    min_chars: int,
    jobs: int | None = None,
    media_kinds: Sequence[str] = (),
) -> tuple[list[Document], list[Skip]]:
    documents: list[Document] = []
    skipped: list[Skip] = []
    media_set = set(media_kinds)

    extracted: dict[Path, tuple[str | None, str | None]] = {}
    heavy = [path for path in files if is_heavy(path)]
    workers = _auto_worker_count(heavy) if not jobs or jobs <= 0 else jobs
    if workers > 1 and len(heavy) >= 2:
        try:
            extracted = _parallel_heavy_results(heavy, max_pdf_pages, workers)
        except Exception:
            LOG.warning("Parallel extraction unavailable; falling back to sequential.")
            LOG.debug("Parallel extraction failure", exc_info=True)
            extracted = {}

    for path in files:
        kind = media_kind(path)
        if kind is not None:
            if kind in media_set:
                documents.append(Document(path, "", [], kind))
            else:
                skipped.append(Skip(path, f"{kind} embedding not enabled"))
            continue

        outcome = extracted.get(path)
        if outcome is None:
            outcome = extract_worker((str(path), max_pdf_pages))

        raw, error = outcome
        if error is not None:
            skipped.append(Skip(path, error))
            continue

        text = " ".join((raw or "").split())
        chunks = split_chunks(text, max_chars, max_chunks)
        if not chunks or len(text) < min_chars:
            skipped.append(Skip(path, "too little text"))
            continue
        documents.append(Document(path, text, chunks))

    return documents, skipped


# --------------------------------------------------------------------------- #
# Embedding backends
# --------------------------------------------------------------------------- #


class Embedder(Protocol):
    name: str
    cacheable: bool
    media_kinds: frozenset[str]

    def embed_corpus(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_media(self, items: Sequence[tuple[Path, str]]) -> np.ndarray: ...


class EmbedderError(Exception):
    """Raised when an embedding backend cannot be constructed."""


def _l2_normalize(matrix) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.size == 0:
        return array
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class TfidfEmbedder:
    """Corpus-local fallback that needs no model download.

    Quality is well below a trained embedding model, but it keeps the tool
    usable out of the box and is fully deterministic.
    """

    cacheable = False
    media_kinds: frozenset[str] = frozenset()

    def __init__(self, max_features: int = 20000) -> None:
        self.name = "tfidf"
        self._max_features = max_features

    def embed_media(self, items: Sequence[tuple[Path, str]]) -> np.ndarray:
        raise NotImplementedError("the TF-IDF backend cannot embed media")

    def embed_corpus(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self._max_features,
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary (e.g. only stop words): fall back to characters.
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=self._max_features
            )
            matrix = vectorizer.fit_transform(texts)
        return _l2_normalize(np.asarray(matrix.todense(), dtype=np.float32))


class LlamaEmbedder:
    """Local GGUF embedding model through llama-cpp-python."""

    cacheable = True
    media_kinds: frozenset[str] = frozenset()

    def __init__(self, model_path: Path, n_ctx: int = 2048) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbedderError(
                "llama-cpp-python is not installed. Run "
                "`pip install llama-cpp-python`, or use --backend tfidf."
            ) from exc

        if not Path(model_path).is_file():
            raise EmbedderError(
                f"Embedding model not found: {model_path}\n"
                "Download a GGUF embedding model (for example EmbeddingGemma) "
                "and pass --model PATH."
            )

        try:
            self._model = Llama(
                model_path=str(model_path), embedding=True, n_ctx=n_ctx, verbose=False
            )
        except Exception as exc:  # pragma: no cover - depends on model file
            raise EmbedderError(f"Failed to load GGUF model {model_path}: {exc}") from exc

        self.name = f"gguf:{Path(model_path).name}:ctx{n_ctx}"

    def embed_corpus(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        total = len(texts)
        for index, text in enumerate(texts, start=1):
            try:
                response = self._model.create_embedding(text)
                vectors.append(response["data"][0]["embedding"])
            except Exception as exc:  # pragma: no cover - depends on model file
                raise RuntimeError(f"Embedding failed on chunk {index}/{total}: {exc}") from exc
            if index % 25 == 0 or index == total:
                LOG.info("Embedded %d/%d chunks", index, total)

        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))

    def embed_media(self, items: Sequence[tuple[Path, str]]) -> np.ndarray:
        raise NotImplementedError("the GGUF text backend cannot embed media")


def _prepare_windows_ffmpeg() -> None:
    """Make FFmpeg shared libraries visible to TorchCodec on Windows.

    TorchCodec requires FFmpeg DLLs; PyAV already ships them. Copy PyAV's
    libraries to a cache directory under the unhashed names TorchCodec looks
    for, once per PyAV version, then expose that directory.
    """
    if os.name != "nt":
        return
    try:
        import av
    except ImportError:
        return

    bundled = Path(av.__file__).parent.parent / "av.libs"
    if not bundled.is_dir():
        return

    cache_root = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    target = cache_root / "smart-sort" / f"ffmpeg-{av.__version__}"
    marker = target / ".ready"
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if marker.exists() and str(target) in path_entries:
        return

    if not marker.exists():
        try:
            target.mkdir(parents=True, exist_ok=True)
            for dll in bundled.glob("*.dll"):
                shutil.copy2(dll, target / dll.name)
                match = re.match(r"^(.+?-\d+)-[0-9a-f]{32}\.dll$", dll.name)
                if match:
                    shutil.copy2(dll, target / f"{match.group(1)}.dll")
            dummy = target / "ffmpeg.exe"
            if not dummy.exists():
                dummy.write_bytes(b"MZ")
            marker.write_text("ok", encoding="utf-8")
        except OSError as exc:
            LOG.warning("Could not prepare FFmpeg libraries: %s", exc)
            return

    if str(target) not in path_entries:
        os.environ["PATH"] = str(target) + os.pathsep + os.environ.get("PATH", "")
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(str(target))
        except OSError:
            pass
    LOG.debug("Exposed PyAV FFmpeg libraries from %s", target)


class SentenceTransformerEmbedder:
    """Multimodal backend (EmbeddingGemma 2) via sentence-transformers.

    Text, images, audio, and video share one vector space, so a photo of an
    invoice and an invoice PDF can land in the same cluster.
    """

    cacheable = True

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        batch_size: int = 32,
        text_prompt: str | None = None,
        media: Sequence[str] = ("image", "audio", "video"),
    ) -> None:
        _prepare_windows_ffmpeg()
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbedderError(
                "sentence-transformers is not installed. Run "
                "`pip install -r requirements-multimodal.txt`."
            ) from exc

        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype == "auto":
            resolved_dtype = (
                torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
            )
        else:
            dtypes = {
                "bf16": torch.bfloat16,
                "bfloat16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
                "fp32": torch.float32,
                "float32": torch.float32,
            }
            if dtype not in dtypes:
                raise EmbedderError(f"Unsupported dtype: {dtype}")
            resolved_dtype = dtypes[dtype]
        if resolved_dtype == torch.float16:
            LOG.warning("EmbeddingGemma 2 returns NaN in fp16; prefer bf16 or fp32.")

        try:
            self._model = SentenceTransformer(
                model_name,
                device=resolved_device,
                model_kwargs={"torch_dtype": resolved_dtype},
            )
        except Exception as exc:  # pragma: no cover - depends on model files
            raise EmbedderError(f"Failed to load {model_name}: {exc}") from exc

        prompts = getattr(self._model, "prompts", None) or {}
        self._prompt_name = text_prompt if text_prompt in prompts else None
        if text_prompt and self._prompt_name is None:
            LOG.warning(
                "Prompt %r is not defined by %s; encoding text without a prompt.",
                text_prompt,
                model_name,
            )
        elif self._prompt_name is None and "Document" in prompts:
            self._prompt_name = "Document"

        self._batch_size = max(1, batch_size)
        self.media_kinds = frozenset(media)
        dtype_name = str(resolved_dtype).replace("torch.", "")
        prompt_name = self._prompt_name or "none"
        self.name = f"st:{model_name}:{resolved_device}:{dtype_name}:{prompt_name}"

    def _encode(self, inputs, **kwargs) -> np.ndarray:
        vectors = self._model.encode(
            inputs,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            **kwargs,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_corpus(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        kwargs = {"prompt_name": self._prompt_name} if self._prompt_name else {}
        return _l2_normalize(self._encode(list(texts), **kwargs))

    def embed_media(self, items: Sequence[tuple[Path, str]]) -> np.ndarray:
        if not items:
            return np.zeros((0, 1), dtype=np.float32)
        payload = []
        for path, kind in items:
            if kind == "image":
                payload.append({"text": "<|image|>", "image": [str(path)]})
            else:
                payload.append({"text": f"<|{kind}|>", kind: str(path)})
        return _l2_normalize(self._encode(payload))


def build_embedder(settings: Settings) -> Embedder:
    if settings.backend == "tfidf":
        LOG.info("Using the TF-IDF backend (no model required).")
        return TfidfEmbedder()

    if settings.backend == "st":
        return SentenceTransformerEmbedder(
            settings.model_path or "unsloth/embeddinggemma-2",
            device=settings.device,
            dtype=settings.dtype,
            batch_size=settings.batch_size,
            text_prompt=settings.text_prompt,
            media=settings.media,
        )

    candidate = Path(settings.model_path) if settings.model_path else None
    if settings.backend == "gguf":
        if candidate is None:
            raise EmbedderError("--backend gguf requires --model PATH.")
        return LlamaEmbedder(candidate, settings.n_ctx)

    # auto: prefer a GGUF model when one is explicitly given or present locally.
    default_model = Path("embeddinggemma-2.gguf")
    model = candidate or default_model
    if model.is_file():
        try:
            return LlamaEmbedder(model, settings.n_ctx)
        except EmbedderError as exc:
            LOG.warning("Could not use %s (%s); falling back to TF-IDF.", model, exc)

    LOG.warning(
        "No GGUF model found; using the TF-IDF backend. "
        "Pass --model PATH for better semantic quality."
    )
    return TfidfEmbedder()


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #


def _slug(value: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug[:max_length]


class EmbeddingCache:
    """Persist chunk vectors so re-runs do not re-embed unchanged text."""

    def __init__(self, path: Path, identity: dict) -> None:
        self.path = path
        self.identity = identity
        self._vectors: dict[str, np.ndarray] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        try:
            with np.load(self.path, allow_pickle=False) as data:
                meta = json.loads(str(data["__meta__"][0])) if "__meta__" in data.files else {}
                if meta != self.identity:
                    return
                for key in data.files:
                    if key != "__meta__":
                        self._vectors[key] = np.asarray(data[key], dtype=np.float32)
        except Exception:
            LOG.debug("Ignoring unreadable embedding cache %s", self.path, exc_info=True)
            self._vectors = {}

    def get(self, key: str) -> np.ndarray | None:
        return self._vectors.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = np.asarray(vector, dtype=np.float32)

    def save(self) -> None:
        if not self._vectors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp.npz")
        payload = {"__meta__": np.array([json.dumps(self.identity)])}
        payload.update(self._vectors)
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, self.path)


def _media_cache_key(path: Path, kind: str) -> str:
    try:
        stats = path.stat()
        return f"media:{kind}:{path.as_posix()}:{stats.st_size}:{stats.st_mtime_ns}"
    except OSError:
        return f"media:{kind}:{path.as_posix()}"


def _embed_media_with_fallback(
    embedder: Embedder, items: Sequence[tuple[Path, str]], batch_size: int = 8
) -> tuple[dict[Path, np.ndarray], list[Path]]:
    """Embed media in batches, retrying failures individually to isolate bad files."""
    embedded: dict[Path, np.ndarray] = {}
    failed: list[Path] = []

    for start in range(0, len(items), batch_size):
        batch = list(items[start : start + batch_size])
        try:
            fresh = embedder.embed_media(batch)
            if len(fresh) != len(batch):
                raise RuntimeError(
                    f"embedder returned {len(fresh)} vectors for {len(batch)} media items"
                )
            for (path, _kind), vector in zip(batch, fresh):
                embedded[path] = np.asarray(vector, dtype=np.float32)
            continue
        except Exception as exc:
            LOG.warning("Media batch failed (%s); retrying individually.", exc)

        for item in batch:
            try:
                single = embedder.embed_media([item])
                if len(single) != 1:
                    raise RuntimeError("embedder returned an unexpected vector count")
                embedded[item[0]] = np.asarray(single[0], dtype=np.float32)
            except Exception as exc:
                failed.append(item[0])
                LOG.warning("Cannot embed %s: %s", item[0].name, exc)

    return embedded, failed


def embed_documents(
    documents: Sequence[Document],
    embedder: Embedder,
    cache_dir: Path | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, list[Document], list[Skip]]:
    """Embed text and media, returning the matrix and the documents that succeeded."""
    cache = None
    if cache_dir is not None and use_cache and embedder.cacheable:
        cache_path = cache_dir / f"embeddings-{_slug(embedder.name)}.npz"
        cache = EmbeddingCache(cache_path, {"embedder": embedder.name})

    vectors: dict[int, np.ndarray] = {}
    failures: dict[int, str] = {}

    # ---- text chunks ------------------------------------------------------ #
    chunk_positions: dict[str, int] = {}
    chunk_texts: list[str] = []
    document_chunks: dict[int, list[int]] = {}

    for index, document in enumerate(documents):
        if document.kind != "text":
            continue
        positions: list[int] = []
        for chunk in document.chunks:
            key = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            position = chunk_positions.get(key)
            if position is None:
                position = len(chunk_texts)
                chunk_positions[key] = position
                chunk_texts.append(chunk)
            positions.append(position)
        document_chunks[index] = positions

    keys = list(chunk_positions)
    chunk_vectors: list[np.ndarray | None] = [None] * len(keys)
    missing_chunks: list[int] = []
    for index, key in enumerate(keys):
        cached = cache.get(key) if cache else None
        if cached is None:
            missing_chunks.append(index)
        else:
            chunk_vectors[index] = cached

    if missing_chunks:
        LOG.info(
            "Embedding %d chunks (%d cached)",
            len(missing_chunks),
            len(keys) - len(missing_chunks),
        )
        fresh = embedder.embed_corpus([chunk_texts[index] for index in missing_chunks])
        if len(fresh) != len(missing_chunks):
            raise RuntimeError(
                f"Embedder returned {len(fresh)} vectors for {len(missing_chunks)} chunks"
            )
        for index, vector in zip(missing_chunks, fresh):
            chunk_vectors[index] = np.asarray(vector, dtype=np.float32)
            if cache:
                cache.put(keys[index], chunk_vectors[index])
        if cache:
            cache.save()
    elif keys:
        LOG.info("All %d chunks loaded from cache", len(keys))

    for index, positions in document_chunks.items():
        if positions:
            vectors[index] = np.mean([chunk_vectors[p] for p in positions], axis=0)
        else:
            failures[index] = "no text chunks"

    # ---- media ------------------------------------------------------------ #
    media_indices = [index for index, doc in enumerate(documents) if doc.kind != "text"]
    missing_media: list[tuple[int, str]] = []
    for index in media_indices:
        document = documents[index]
        key = _media_cache_key(document.path, document.kind)
        cached = cache.get(key) if cache else None
        if cached is None:
            missing_media.append((index, key))
        else:
            vectors[index] = cached

    if missing_media:
        if not embedder.media_kinds:
            for index, _key in missing_media:
                failures[index] = "media not supported by this backend"
        else:
            items_by_kind: dict[str, list[tuple[Path, str]]] = {}
            for index, _key in missing_media:
                document = documents[index]
                items_by_kind.setdefault(document.kind, []).append(
                    (document.path, document.kind)
                )

            embedded: dict[Path, np.ndarray] = {}
            failed_paths: list[Path] = []
            for kind, items in items_by_kind.items():
                LOG.info("Embedding %d %s files", len(items), kind)
                kind_vectors, kind_failures = _embed_media_with_fallback(embedder, items)
                embedded.update(kind_vectors)
                failed_paths.extend(kind_failures)

            failed_set = set(failed_paths)
            for index, key in missing_media:
                document = documents[index]
                if document.path in failed_set:
                    failures[index] = f"{document.kind} could not be decoded"
                    continue
                vectors[index] = embedded[document.path]
                if cache:
                    cache.put(key, vectors[index])
            if cache:
                cache.save()
    elif media_indices:
        LOG.info("All %d media files loaded from cache", len(media_indices))

    # ---- assemble --------------------------------------------------------- #
    dims = {vector.shape[0] for vector in vectors.values()}
    if len(dims) > 1:
        raise RuntimeError(f"Inconsistent embedding dimensions: {sorted(dims)}")
    dimension = dims.pop() if dims else 1

    kept: list[Document] = []
    skipped: list[Skip] = []
    rows: list[np.ndarray] = []
    for index, document in enumerate(documents):
        if index in failures:
            skipped.append(Skip(document.path, failures[index]))
            continue
        vector = vectors.get(index)
        if vector is None:
            skipped.append(Skip(document.path, "no embedding produced"))
            continue
        kept.append(document)
        rows.append(vector)

    if not rows:
        return np.zeros((0, dimension), dtype=np.float32), kept, skipped
    return _l2_normalize(np.vstack(rows)), kept, skipped


# --------------------------------------------------------------------------- #
# Clustering and naming
# --------------------------------------------------------------------------- #


def choose_k(
    count: int,
    requested: int | None = None,
    auto: bool = False,
    seed: int = 0,
    matrix: np.ndarray | None = None,
) -> int:
    """Resolve k: explicit value, silhouette search, or sqrt(n/2) heuristic."""
    if count <= 1:
        return max(1, count)
    if requested is not None:
        if requested < 1:
            raise ValueError("k must be at least 1")
        if requested > count:
            LOG.warning("Requested k=%d exceeds file count %d; using %d.", requested, count, count)
            requested = count
        return requested

    if auto and matrix is not None and count >= 4:
        best_k: int | None = None
        best_score = -2.0
        for candidate in range(2, min(10, count - 1) + 1):
            try:
                labels = KMeans(
                    n_clusters=candidate, n_init=10, random_state=seed
                ).fit_predict(matrix)
                if len(set(labels.tolist())) < 2:
                    continue
                score = float(silhouette_score(matrix, labels))
            except Exception:
                continue
            if score > best_score:
                best_k, best_score = candidate, score
        if best_k is not None:
            LOG.info("Silhouette search chose k=%d (score %.3f)", best_k, best_score)
            return best_k

    return max(2, min(20, count, round(math.sqrt(count / 2))))


def build_cluster_names(documents: Sequence[Document], labels: np.ndarray) -> dict[int, str]:
    """Name folders ``Cluster_<n>_<keywords>`` from distinctive TF-IDF terms."""
    terms_by_cluster: dict[int, list[str]] = {}

    if documents:
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20000,
                sublinear_tf=True,
            )
            matrix = np.asarray(
                vectorizer.fit_transform([d.text[:4000] for d in documents]).todense(),
                dtype=np.float32,
            )
            vocabulary = np.array(vectorizer.get_feature_names_out())
            for cluster in np.unique(labels):
                mask = labels == cluster
                centroid = matrix[mask].mean(axis=0)
                others = matrix[~mask].mean(axis=0) if (~mask).any() else np.zeros_like(centroid)
                scores = (centroid - others) * (centroid > 0)
                top = np.argsort(scores)[-4:][::-1]
                terms_by_cluster[int(cluster)] = [
                    str(vocabulary[i]) for i in top if scores[i] > 0
                ]
        except ValueError:
            terms_by_cluster = {}

    names: dict[int, str] = {}
    used: set[str] = set()
    for cluster in np.unique(labels):
        cluster = int(cluster)
        base = f"Cluster_{cluster + 1}"
        keywords = _slug(" ".join(terms_by_cluster.get(cluster, [])[:2]))[:40].strip("_")
        name = f"{base}_{keywords}" if keywords else base
        if name.casefold() in used:
            name = f"{name}_{cluster + 1}"
        used.add(name.casefold())
        names[cluster] = name
    return names


MEDIA_LABELS: tuple[str, ...] = (
    "photographs",
    "screenshots",
    "scanned documents",
    "receipts",
    "invoices",
    "presentations",
    "spreadsheets",
    "charts and diagrams",
    "maps",
    "logos and icons",
    "wallpapers",
    "social media images",
    "memes",
    "product photos",
    "people and portraits",
    "landscapes",
    "food",
    "buildings",
    "artwork",
    "music",
    "speech recordings",
    "podcasts",
    "videos",
    "screen recordings",
    "movies and clips",
)


def label_media_clusters(
    documents: Sequence[Document],
    labels: np.ndarray,
    matrix: np.ndarray,
    embedder: Embedder,
    names: dict[int, str],
) -> dict[int, str]:
    """Name media-only clusters by nearest text label in the shared space."""
    if not getattr(embedder, "media_kinds", frozenset()) or not len(labels):
        return names

    has_text = {int(label): False for label in np.unique(labels)}
    for document, label in zip(documents, labels):
        if document.kind == "text" and document.text:
            has_text[int(label)] = True
    media_clusters = [cluster for cluster, present in has_text.items() if not present]
    if not media_clusters:
        return names

    try:
        label_vectors = _l2_normalize(embedder.embed_corpus(list(MEDIA_LABELS)))
    except Exception as exc:  # pragma: no cover - depends on backend
        LOG.debug("Media cluster naming unavailable: %s", exc)
        return names

    for cluster in media_clusters:
        centroid = matrix[labels == cluster].mean(axis=0)
        centroid = _l2_normalize(centroid.reshape(1, -1))[0]
        best = int(np.argmax(label_vectors @ centroid))
        names[cluster] = f"Cluster_{cluster + 1}_{_slug(MEDIA_LABELS[best])}"
    return names


# --------------------------------------------------------------------------- #
# Planning, applying and undoing moves
# --------------------------------------------------------------------------- #


def _unique_destination(directory: Path, filename: str, used: set[str]) -> Path:
    stem, suffix = Path(filename).stem, Path(filename).suffix
    candidate = directory / filename
    counter = 2
    while str(candidate).casefold() in used or candidate.exists():
        candidate = directory / f"{stem} ({counter}){suffix}"
        counter += 1
    return candidate


def plan_moves(
    root: Path,
    documents: Sequence[Document],
    labels: np.ndarray,
    names: dict[int, str],
) -> list[Move]:
    moves: list[Move] = []
    used: set[str] = set()

    for document, label in zip(documents, labels):
        cluster = int(label)
        destination = _unique_destination(root / names[cluster], document.path.name, used)
        used.add(str(destination).casefold())
        moves.append(Move(document.path, destination, cluster))

    return moves


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_moves(root: Path, moves: Sequence[Move], metadata: dict) -> ApplyResult:
    root = root.resolve()
    state_dir = root / STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    manifest = state_dir / f"manifest-{stamp}.jsonl"
    if manifest.exists():
        manifest = state_dir / f"manifest-{stamp}-{os.getpid()}.jsonl"

    moved = 0
    failed = 0
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "header",
                    "version": 1,
                    "created": datetime.now(timezone.utc).isoformat(),
                    "root": str(root),
                    **metadata,
                }
            )
            + "\n"
        )
        handle.flush()

        for move in moves:
            try:
                if move.dst.exists():
                    raise OSError(f"destination already exists: {move.dst}")
                size = move.src.stat().st_size
                digest = sha256_file(move.src)
                move.dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(move.src), str(move.dst))
                if not move.dst.exists():
                    raise OSError("destination missing after move")
                handle.write(
                    json.dumps(
                        {
                            "type": "move",
                            "src": move.src.relative_to(root).as_posix(),
                            "dst": move.dst.relative_to(root).as_posix(),
                            "size": size,
                            "sha256": digest,
                        }
                    )
                    + "\n"
                )
                handle.flush()
                moved += 1
            except OSError as exc:
                LOG.error("Could not move %s: %s", move.src, exc)
                failed += 1

    if moved == 0:
        manifest.unlink(missing_ok=True)
        return ApplyResult(0, failed, None)

    return ApplyResult(moved, failed, manifest)


def _restored_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    candidate = parent / f"{stem} (restored){suffix}"
    counter = 2
    while candidate.exists():
        candidate = parent / f"{stem} (restored {counter}){suffix}"
        counter += 1
    return candidate


def _prune_empty_cluster_dirs(root: Path, directories: Sequence[Path]) -> None:
    for directory in sorted(set(directories), key=lambda p: len(p.parts), reverse=True):
        current = directory
        while current != root and current.parent != current:
            if not CLUSTER_DIR_RE.match(current.name):
                break
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent


def undo_manifest(manifest: Path, dry_run: bool = True) -> tuple[int, int]:
    """Move files recorded in a manifest back to their original paths."""
    header: dict | None = None
    entries: list[dict] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                header = record
            elif record.get("type") == "move":
                entries.append(record)

    if header is None:
        raise ValueError(f"{manifest} is not a Smart-Sort manifest (no header)")

    root = Path(header["root"])
    restored = 0
    failed = 0
    touched: list[Path] = []

    for record in reversed(entries):
        source = root / record["src"]
        destination = root / record["dst"]

        if not destination.exists():
            LOG.warning("Missing, cannot undo: %s", record["dst"])
            failed += 1
            continue
        expected = record.get("sha256")
        if expected and sha256_file(destination) != expected:
            LOG.warning("Checksum changed, refusing to move: %s", record["dst"])
            failed += 1
            continue

        target = _restored_path(source)
        if dry_run:
            print(f"  would restore {record['dst']} -> {target.relative_to(root).as_posix()}")
            restored += 1
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(target))
        touched.append(destination.parent)
        restored += 1

    if not dry_run:
        _prune_empty_cluster_dirs(root, touched)

    return restored, failed


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _print_summary(root: Path, moves: Sequence[Move], settings: Settings) -> None:
    by_folder: dict[str, list[Move]] = {}
    for move in moves:
        by_folder.setdefault(move.dst.parent.name, []).append(move)

    print(f"{len(moves)} files -> {len(by_folder)} {'folder' if len(by_folder) == 1 else 'folders'}")
    for folder in sorted(by_folder):
        group = by_folder[folder]
        print(f"  {folder:<44} {len(group):>4} files")
        if settings.verbose_moves:
            for move in sorted(group, key=lambda m: m.src.name):
                try:
                    label = move.src.relative_to(root).as_posix()
                except ValueError:
                    label = str(move.src)
                print(f"      {label}")
        else:
            sample = ", ".join(sorted(move.src.name for move in group)[:3])
            print(f"      {sample}")


def run(settings: Settings, embedder: Embedder | None = None) -> int:
    root = settings.root.resolve()
    watch = Stopwatch()

    with watch.phase("collect"):
        files = collect_files(root, settings.recursive, settings.excludes)
    if not files:
        LOG.error("No eligible files found in %s", root)
        return 1

    try:
        with watch.phase("model"):
            embedder = embedder or build_embedder(settings)
    except EmbedderError as exc:
        LOG.error("%s", exc)
        return 1

    media_kinds = tuple(
        kind for kind in settings.media if kind in getattr(embedder, "media_kinds", frozenset())
    )
    with watch.phase("extract"):
        documents, skipped = load_documents(
            files,
            settings.max_chars,
            settings.max_chunks,
            settings.max_pdf_pages,
            settings.min_chars,
            settings.jobs,
            media_kinds,
        )

    cache_dir = root / STATE_DIR if settings.use_cache else None
    try:
        with watch.phase("embed"):
            matrix, documents, media_skips = embed_documents(
                documents, embedder, cache_dir, settings.use_cache
            )
            skipped.extend(media_skips)
    except (RuntimeError, ValueError) as exc:
        LOG.error("Embedding failed: %s", exc)
        return 1

    if len(documents) < 2:
        LOG.error(
            "Need at least 2 embeddable files, found %d (skipped %d).",
            len(documents),
            len(skipped),
        )
        for skip in skipped:
            LOG.info("Skipped %s: %s", skip.path.name, skip.reason)
        return 1

    with watch.phase("cluster"):
        k = choose_k(len(documents), settings.k, settings.auto_k, settings.seed, matrix)
        k = max(1, min(k, len(documents)))
        with warnings.catch_warnings():
            # Duplicate near-identical documents legitimately yield fewer clusters.
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = KMeans(
                n_clusters=k, n_init=10, random_state=settings.seed
            ).fit_predict(matrix)

        names = (
            build_cluster_names(documents, labels)
            if settings.label_clusters
            else {int(label): f"Cluster_{int(label) + 1}" for label in np.unique(labels)}
        )
        if settings.label_clusters:
            names = label_media_clusters(documents, labels, matrix, embedder, names)
        moves = plan_moves(root, documents, labels, names)

    _print_summary(root, moves, settings)

    if skipped:
        counts = Counter(skip.reason for skip in skipped)
        detail = ", ".join(f"{reason} x{count}" for reason, count in counts.most_common())
        LOG.warning("Skipped %d files: %s", len(skipped), detail)
        for skip in skipped:
            LOG.debug("Skipped %s: %s", skip.path, skip.reason)

    if not settings.apply:
        if settings.timings:
            watch.print_summary()
        print("\nDry run: no files were moved. Re-run with --apply to execute.")
        return 0

    with watch.phase("apply"):
        result = apply_moves(
            root,
            moves,
            {
                "backend": embedder.name,
                "k": k,
                "files": len(documents),
                "media": ",".join(media_kinds),
                "seed": settings.seed,
            },
        )
    if result.manifest is None:
        LOG.error("No files were moved (%d failed).", result.failed)
        return 1

    relative_manifest = result.manifest.relative_to(root).as_posix()
    folder_word = "folder" if len(names) == 1 else "folders"
    print(f"\nMoved {result.moved} files into {len(names)} {folder_word}.")
    if result.failed:
        print(f"Failed: {result.failed} files (see log above).")
    print(f"Manifest: {relative_manifest}")
    print(f'Undo with: python smart_sort.py --undo "{relative_manifest}" --apply')
    if settings.timings:
        watch.print_summary()
    return 0 if result.failed == 0 else 1


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart_sort.py",
        description="Organize a folder into semantic clusters based on file content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            '  smart_sort.py "D:\\Downloads"                     # preview (safe)\n'
            '  smart_sort.py "D:\\Downloads" --apply             # reorganize\n'
            '  smart_sort.py . --backend tfidf --k 6 --recursive\n'
            '  smart_sort.py "D:\\Media" --backend st --model unsloth/embeddinggemma-2\n'
            '  smart_sort.py --model models/embeddinggemma-2.gguf --apply "D:\\Docs"\n'
            '  smart_sort.py --undo ".smart_sort/manifest-20250101-120000.jsonl" --apply\n'
        ),
    )
    parser.add_argument("path", nargs="?", help="Folder to organize.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag the run is a dry run.",
    )
    parser.add_argument(
        "--undo",
        metavar="MANIFEST",
        help="Reverse a previous run using its JSONL manifest (dry run unless --apply).",
    )
    parser.add_argument("--k", type=int, help="Number of clusters (default: automatic).")
    parser.add_argument(
        "--auto-k",
        action="store_true",
        help="Choose k by silhouette score instead of the sqrt(n/2) heuristic.",
    )
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Include files in subfolders."
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "gguf", "tfidf", "st"),
        default="auto",
        help=(
            "Embedding backend: st = sentence-transformers (multimodal "
            "EmbeddingGemma 2), gguf = llama.cpp text model, tfidf = no model, "
            "auto = GGUF if present else TF-IDF."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        help=(
            "GGUF file path (gguf backend) or Hugging Face id / local path "
            "(st backend, default: unsloth/embeddinggemma-2)."
        ),
    )
    parser.add_argument(
        "--media",
        default="image,audio,video",
        help="Media kinds to embed with the st backend (default: image,audio,video).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for the st backend: auto, cpu, cuda, cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype for the st backend: auto, bf16, fp32 (default: auto).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the st backend (default: 32).",
    )
    parser.add_argument(
        "--text-prompt",
        default=None,
        help="Task prompt name for text embeddings (e.g. Document, SearchQuery).",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=2048, help="Context size for the GGUF model."
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Characters per embedded chunk (default: 4000).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=4,
        help="Maximum chunks sampled per document (default: 4).",
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=20,
        help="Pages to read per PDF (default: 20).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=20,
        help="Skip files with less text than this (default: 20).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Worker processes for PDF/DOCX extraction: 0 = auto "
            "(only for large batches), 1 = sequential, N = force N workers."
        ),
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="Print per-phase timing information.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Use plain Cluster_N folder names instead of keyword labels.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write the embedding cache.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="PATTERN",
        help="Extra file/folder name glob to skip (repeatable).",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="List every planned move."
    )
    return parser


def _configure_output_streams() -> None:
    """Never crash printing names the console encoding cannot represent."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass


def main(argv: Sequence[str] | None = None) -> int:
    _configure_output_streams()
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    for noisy in (
        "pypdf",
        "PyPDF2",
        "httpx",
        "huggingface_hub",
        "filelock",
        "urllib3",
        "sentence_transformers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.undo:
        manifest = Path(args.undo).expanduser()
        if not manifest.is_file():
            parser.error(f"manifest not found: {manifest}")
        started = time.perf_counter()
        try:
            restored, failed = undo_manifest(manifest, dry_run=not args.apply)
        except (OSError, ValueError) as exc:
            LOG.error("%s", exc)
            return 1
        elapsed = time.perf_counter() - started
        verb = "Restored" if args.apply else "Would restore"
        print(f"{verb} {restored} files.")
        if args.timings:
            print(f"\nTimings:\n  {'undo':<12} {elapsed:8.3f} s")
        if not args.apply:
            print("Dry run: no files were moved. Re-run with --apply to execute.")
        if failed:
            print(f"Failed: {failed} files (see log above).")
            return 1
        return 0

    if not args.path:
        parser.error("path is required unless --undo is used")

    root = Path(args.path).expanduser()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    media = tuple(part.strip().lower() for part in args.media.split(",") if part.strip())
    unknown_media = set(media) - {"image", "audio", "video"}
    if unknown_media:
        parser.error(f"unknown media kinds: {', '.join(sorted(unknown_media))}")

    settings = Settings(
        root=root,
        k=args.k,
        auto_k=args.auto_k,
        apply=args.apply,
        recursive=args.recursive,
        backend=args.backend,
        model_path=args.model,
        n_ctx=args.n_ctx,
        media=media,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        text_prompt=args.text_prompt,
        max_chars=args.max_chars,
        max_chunks=args.max_chunks,
        max_pdf_pages=args.max_pdf_pages,
        min_chars=args.min_chars,
        seed=args.seed,
        label_clusters=not args.no_labels,
        use_cache=not args.no_cache,
        jobs=args.jobs if args.jobs > 0 else None,
        timings=args.timings,
        verbose_moves=bool(args.verbose),
        excludes=DEFAULT_EXCLUDES + tuple(args.exclude or ()),
    )

    try:
        return run(settings)
    except KeyboardInterrupt:
        LOG.warning("Interrupted before completion; any completed moves are in the manifest.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
