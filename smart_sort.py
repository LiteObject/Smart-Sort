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
import shlex
import shutil
import stat
import sys
import tempfile
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from file_safety import (
    CLUSTER_DIR_RE,
    STATE_DIR,
    ApplyResult,
    Move,
    apply_moves,
    assert_safe_path,
    is_link_or_reparse,
    resolve_root,
    safe_relative_path,
    sha256_file,
    undo_manifest,
)
from text_extract import ExtractionError, extract_text, extract_worker, is_heavy, looks_like_text

LOG = logging.getLogger("smart-sort")

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
    organize: bool = False
    source: Path | None = None
    match_threshold: float | None = None
    match_margin: float | None = None
    new_group_threshold: float | None = None
    min_group_size: int = 2
    save_plan: Path | None = None


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
    """Return candidate files, deterministically ordered, links excluded."""
    resolve_root(root)
    files: list[Path] = []

    def consider(path: Path) -> None:
        if is_link_or_reparse(path) or not path.is_file() or is_hidden_or_system(path):
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
                and not is_link_or_reparse(Path(dirpath) / d)
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
            reason = (
                "no extractable text (possibly a scanned PDF)"
                if path.suffix.lower() == ".pdf" and not text else "too little text"
            )
            skipped.append(Skip(path, reason))
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


def _valid_embedding(vector: np.ndarray) -> bool:
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        return False
    norm = float(np.linalg.norm(vector))
    return math.isfinite(norm) and norm > 0


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
            dtype=np.float32,
        )
        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary (e.g. only stop words): fall back to characters.
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), max_features=self._max_features,
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform(texts)
        return _l2_normalize(matrix.toarray())


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
            before = Path(model_path).stat()
            fingerprint = sha256_file(Path(model_path))
            after = Path(model_path).stat()
            if (
                not os.path.samestat(before, after)
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise EmbedderError(f"GGUF model changed while hashing: {model_path}")
        except OSError as exc:
            raise EmbedderError(f"Cannot fingerprint GGUF model {model_path}: {exc}") from exc

        try:
            self._model = Llama(
                model_path=str(model_path), embedding=True, n_ctx=n_ctx, verbose=False
            )
        except Exception as exc:  # pragma: no cover - depends on model file
            raise EmbedderError(f"Failed to load GGUF model {model_path}: {exc}") from exc

        try:
            loaded = Path(model_path).stat()
        except OSError as exc:
            raise EmbedderError(f"Cannot verify GGUF model {model_path}: {exc}") from exc
        if (
            not os.path.samestat(after, loaded)
            or after.st_size != loaded.st_size
            or after.st_mtime_ns != loaded.st_mtime_ns
        ):
            raise EmbedderError(f"GGUF model changed while loading: {model_path}")
        self.name = f"gguf:{Path(model_path).name}:{fingerprint}:ctx{n_ctx}"

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
            if resolved_device.startswith("cuda"):
                try:
                    with torch.cuda.device(resolved_device):
                        supports_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                except (RuntimeError, ValueError) as exc:
                    raise EmbedderError(f"Cannot inspect CUDA device {resolved_device}: {exc}") from exc
                resolved_dtype = torch.bfloat16 if supports_bf16 else torch.float32
            else:
                resolved_dtype = torch.float32
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
        modules = list(self._model.children())
        first = modules[0] if modules else None
        config = getattr(getattr(first, "auto_model", None), "config", None)
        revision = getattr(config, "_commit_hash", None)
        self.cacheable = (
            not Path(model_name).exists()
            and isinstance(revision, str)
            and re.fullmatch(r"[0-9a-f]{40,64}", revision) is not None
        )
        if not self.cacheable:
            LOG.warning(
                "Embedding cache disabled for %s: an immutable remote model revision "
                "could not be verified.", model_name,
            )
        cache_revision = revision if self.cacheable else "uncacheable"
        self.name = f"st:{model_name}:{cache_revision}:{resolved_device}:{dtype_name}:{prompt_name}"

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
        self.root = resolve_root(path.parent.parent)
        assert_safe_path(self.root, self.path, allow_missing=True)
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

    def discard(self, key: str) -> None:
        self._vectors.pop(key, None)

    def save(self) -> None:
        if not self._vectors:
            return
        assert_safe_path(self.root, self.path, allow_missing=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        assert_safe_path(self.root, self.path.parent)
        payload = {"__meta__": np.array([json.dumps(self.identity)])}
        payload.update(self._vectors)
        temporary: Path | None = None
        created: os.stat_result | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", prefix=".embeddings-", suffix=".npz",
                dir=self.path.parent, delete=False,
            ) as handle:
                temporary = Path(handle.name)
                created = os.fstat(handle.fileno())
                np.savez_compressed(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            assert_safe_path(self.root, temporary)
            assert_safe_path(self.root, self.path, allow_missing=True)
            os.replace(temporary, self.path)
        finally:
            if temporary is not None and created is not None and temporary.exists():
                assert_safe_path(self.root, temporary)
                if os.path.samestat(created, temporary.stat()):
                    temporary.unlink()


def _media_cache_key(path: Path, kind: str) -> str:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        not os.path.samestat(before, after)
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise OSError(f"Media file changed while hashing: {path}")
    return f"media:{kind}:{digest}"


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
        if cache is not None and cached is not None and not _valid_embedding(cached):
            LOG.warning("Ignoring invalid cached text embedding")
            cache.discard(key)
            cached = None
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
            fresh_vector = np.asarray(vector, dtype=np.float32)
            chunk_vectors[index] = fresh_vector
            if cache and _valid_embedding(fresh_vector):
                cache.put(keys[index], fresh_vector)
        if cache:
            cache.save()
    elif keys:
        LOG.info("All %d chunks loaded from cache", len(keys))

    for index, positions in document_chunks.items():
        if positions:
            parts: list[np.ndarray] = []
            for position in positions:
                part = chunk_vectors[position]
                if part is None or not _valid_embedding(part):
                    failures[index] = "non-finite, zero, or invalid text chunk embedding"
                    break
                parts.append(part)
            else:
                vectors[index] = np.mean(parts, axis=0)
        else:
            failures[index] = "no text chunks"

    # ---- media ------------------------------------------------------------ #
    media_indices = [index for index, doc in enumerate(documents) if doc.kind != "text"]
    missing_media: list[tuple[int, str]] = []
    for index in media_indices:
        document = documents[index]
        if cache is None:
            missing_media.append((index, ""))
            continue
        try:
            key = _media_cache_key(document.path, document.kind)
        except OSError as exc:
            failures[index] = f"cannot fingerprint {document.kind}"
            LOG.warning("Cannot fingerprint %s: %s", document.path.name, exc)
            continue
        cached = cache.get(key)
        if cache is not None and cached is not None and not _valid_embedding(cached):
            LOG.warning("Ignoring invalid cached embedding for %s", document.path.name)
            cache.discard(key)
            cached = None
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
                if cache and _valid_embedding(vectors[index]):
                    cache.put(key, vectors[index])
            if cache:
                cache.save()
    elif media_indices and not any(index in failures for index in media_indices):
        LOG.info("All %d media files loaded from cache", len(media_indices))

    # ---- assemble --------------------------------------------------------- #
    for index, vector in list(vectors.items()):
        if not _valid_embedding(vector):
            failures[index] = "non-finite, zero, or invalid embedding"
            del vectors[index]
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
                dtype=np.float32,
            )
            matrix = vectorizer.fit_transform([d.text[:4000] for d in documents])
            vocabulary = np.array(vectorizer.get_feature_names_out())
            for cluster in np.unique(labels):
                mask = labels == cluster
                centroid = np.asarray(matrix[mask].mean(axis=0)).ravel()
                others = (
                    np.asarray(matrix[~mask].mean(axis=0)).ravel()
                    if (~mask).any() else np.zeros_like(centroid)
                )
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
    root = resolve_root(root)
    moves: list[Move] = []
    used: set[str] = set()

    for document, label in zip(documents, labels):
        cluster = int(label)
        directory = assert_safe_path(root, root / names[cluster], allow_missing=True)
        destination = _unique_destination(directory, document.path.name, used)
        used.add(str(destination).casefold())
        moves.append(Move(document.path, destination, cluster))

    return moves


# --------------------------------------------------------------------------- #
# Opt-in organizing and reviewed plans
# --------------------------------------------------------------------------- #


_REFERENCE_SAMPLE_LIMIT = 25
_NOVEL_GROUP_LIMIT = 1000


@dataclass
class _OrganizationInputs:
    source: Path
    incoming: list[Path]
    references: dict[Path, list[Path]]
    reference_totals: dict[Path, int]


@dataclass
class _OrganizationDecision:
    source: Path
    kind: str
    folder: Path | None
    score: float | None
    reason: str
    destination: Path | None = None


def _organization_inputs(
    root: Path,
    source: Path | None,
    recursive: bool,
    excludes: Sequence[str],
    plan_path: Path | None,
) -> _OrganizationInputs:
    if source is None:
        if recursive:
            raise ValueError("--organize --recursive requires an explicit --source folder")
        source_root = root
        source_top = None
    else:
        source = source.expanduser()
        source_root = assert_safe_path(root, source if source.is_absolute() else root / source)
        if source_root == root or not source_root.is_dir():
            raise ValueError("--source must be a folder inside, but not equal to, the root")
        current = root
        for part in source_root.relative_to(root).parts:
            current = current / part
            if _skip_dir(part, excludes) or is_hidden_or_system(current):
                raise ValueError(f"--source contains an excluded or hidden folder: {current}")
        source_top = source_root.relative_to(root).parts[0].casefold()

    incoming = [
        path for path in collect_files(source_root, recursive, excludes) if path != plan_path
    ]
    references: dict[Path, list[Path]] = {}
    totals: dict[Path, int] = {}
    for folder in sorted(root.iterdir()):
        if is_link_or_reparse(folder) or not folder.is_dir():
            continue
        if (
            folder.name.startswith(".")
            or folder.name.casefold() == source_top
            or _matches_any(folder.name, excludes)
            or is_hidden_or_system(folder)
        ):
            continue
        files = [
            path for path in collect_files(folder, recursive=True, excludes=excludes)
            if path != plan_path
        ]
        references[folder] = files[:_REFERENCE_SAMPLE_LIMIT]
        totals[folder] = len(files)
    return _OrganizationInputs(source_root, incoming, references, totals)


def _organization_snapshot(
    root: Path, inputs: _OrganizationInputs, hash_paths: set[Path]
) -> dict:
    def fingerprint(path: Path) -> dict:
        assert_safe_path(root, path)
        before = path.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Input is not a regular file: {path}")
        digest = sha256_file(path) if path in hash_paths else None
        assert_safe_path(root, path)
        after = path.stat()
        if (
            not os.path.samestat(before, after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise OSError(f"Input changed while preparing the plan: {path}")
        return {
            "path": path.relative_to(root).as_posix(),
            "size": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": digest,
        }

    return {
        "incoming": [fingerprint(path) for path in inputs.incoming],
        "categories": [
            {
                "folder": folder.relative_to(root).as_posix(),
                "total": inputs.reference_totals[folder],
                "files": [fingerprint(path) for path in files],
            }
            for folder, files in inputs.references.items()
        ],
    }


def _organization_hashes(
    root: Path, files: Sequence[Path], media_kinds: Sequence[str]
) -> tuple[dict[Path, str], list[Skip]]:
    fingerprints: dict[Path, str] = {}
    skipped: list[Skip] = []
    for path in files:
        try:
            assert_safe_path(root, path)
            kind = media_kind(path)
            if kind is not None and kind not in media_kinds:
                continue
            if kind is None and not is_heavy(path):
                with path.open("rb") as handle:
                    if not looks_like_text(handle.read(8192)):
                        continue
            before = path.stat()
            digest = sha256_file(path)
            assert_safe_path(root, path)
            after = path.stat()
            if (
                not os.path.samestat(before, after)
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise OSError(f"File changed while hashing: {path}")
            fingerprints[path] = digest
        except OSError as exc:
            LOG.warning("Cannot fingerprint %s: %s", path.name, exc)
            skipped.append(Skip(path, "cannot fingerprint file"))
    return fingerprints, skipped


def _snapshot_hashes(snapshot: dict) -> dict[str, str]:
    records = list(snapshot["incoming"])
    for category in snapshot["categories"]:
        records.extend(category["files"])
    return {
        record["path"]: record["sha256"]
        for record in records if record["sha256"] is not None
    }


def _snapshot_metadata(snapshot: dict) -> tuple:
    def items(records: list[dict]) -> tuple:
        return tuple((record["path"], record["size"], record["mtime_ns"]) for record in records)

    return (
        items(snapshot["incoming"]),
        tuple(
            (category["folder"], category["total"], items(category["files"]))
            for category in snapshot["categories"]
        ),
    )


def _organization_decisions(
    root: Path,
    inputs: _OrganizationInputs,
    documents: Sequence[Document],
    matrix: np.ndarray,
    skipped: Sequence[Skip],
    embedder: Embedder,
    settings: Settings,
) -> list[_OrganizationDecision]:
    vectors: dict[Path, np.ndarray] = {}
    reasons = {skip.path: skip.reason for skip in skipped}
    for document, vector in zip(documents, matrix):
        if not _valid_embedding(vector):
            reasons[document.path] = "non-finite or zero embedding"
        else:
            vectors[document.path] = vector

    profiles: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    for folder, files in inputs.references.items():
        rows = [vectors[path] for path in files if path in vectors]
        if not rows:
            LOG.info("Category %s has no usable reference files", folder.name)
            continue
        samples = _l2_normalize(np.vstack(rows))
        centroid = _l2_normalize(samples.mean(axis=0))[0]
        profiles[folder] = (centroid, samples)
        LOG.info("Category %s: %d reference files", folder.name, len(rows))

    by_path = {document.path: document for document in documents}
    decisions: dict[Path, _OrganizationDecision] = {}
    unmatched: list[Path] = []
    for path in inputs.incoming:
        vector = vectors.get(path)
        if vector is None:
            decisions[path] = _OrganizationDecision(
                path, "review", None, None, reasons.get(path, "no usable embedding")
            )
            continue

        scores: list[tuple[float, Path]] = []
        for folder, (centroid, samples) in profiles.items():
            score = max(float(centroid @ vector), float(np.max(samples @ vector)))
            scores.append((float(np.clip(score, -1.0, 1.0)), folder))
        scores.sort(key=lambda item: (-item[0], item[1].name.casefold()))
        if not scores:
            unmatched.append(path)
            continue

        score, folder = scores[0]
        if settings.match_threshold is None or settings.match_margin is None:
            decisions[path] = _OrganizationDecision(
                path, "review", None, score,
                f"nearest folder is {folder.name}; matching thresholds are not configured",
            )
        elif score >= settings.match_threshold:
            margin = score - scores[1][0] if len(scores) > 1 else None
            if margin is not None and margin < settings.match_margin:
                decisions[path] = _OrganizationDecision(
                    path, "review", None, score,
                    f"ambiguous match between {folder.name} and {scores[1][1].name}",
                )
            else:
                decisions[path] = _OrganizationDecision(
                    path, "existing", folder, score, "matches existing folder"
                )
        else:
            unmatched.append(path)

    if (
        settings.new_group_threshold is not None
        and len(unmatched) >= settings.min_group_size
        and len(unmatched) <= _NOVEL_GROUP_LIMIT
    ):
        novel_documents = [by_path[path] for path in unmatched]
        novel_matrix = _l2_normalize(np.vstack([vectors[path] for path in unmatched]))
        similarity = np.clip(novel_matrix @ novel_matrix.T, -1.0, 1.0)
        distances = np.maximum(0.0, 1.0 - similarity)
        np.fill_diagonal(distances, 0.0)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=float(np.nextafter(1.0 - settings.new_group_threshold, np.inf)),
            compute_full_tree=True,
        ).fit_predict(distances)
        names = (
            build_cluster_names(novel_documents, labels)
            if settings.label_clusters
            else {int(label): f"Cluster_{int(label) + 1}" for label in np.unique(labels)}
        )
        if settings.label_clusters:
            names = label_media_clusters(novel_documents, labels, novel_matrix, embedder, names)
        used_names = {path.name.casefold() for path in root.iterdir()}
        for label in sorted(np.unique(labels)):
            positions = np.flatnonzero(labels == label)
            if len(positions) < settings.min_group_size:
                continue
            cohesion = float(np.min(similarity[np.ix_(positions, positions)]))
            if cohesion < settings.new_group_threshold:
                continue
            base = names[int(label)]
            name = base
            suffix = 2
            while name.casefold() in used_names:
                name = f"{base}_{suffix}"
                suffix += 1
            used_names.add(name.casefold())
            for position in positions:
                path = unmatched[int(position)]
                decisions[path] = _OrganizationDecision(
                    path, "new", root / name, cohesion, "cohesive unmatched group"
                )
    elif len(unmatched) > _NOVEL_GROUP_LIMIT:
        LOG.warning(
            "Too many unmatched files for bounded grouping (%d > %d); review them manually",
            len(unmatched),
            _NOVEL_GROUP_LIMIT,
        )

    for path in unmatched:
        if path not in decisions:
            reason = (
                "new-group threshold is not configured"
                if settings.new_group_threshold is None
                else "no sufficiently cohesive new group"
            )
            decisions[path] = _OrganizationDecision(path, "review", None, None, reason)

    used_destinations: set[str] = set()
    ordered = [decisions[path] for path in inputs.incoming]
    for decision in ordered:
        if decision.folder is not None:
            folder = assert_safe_path(root, decision.folder, allow_missing=True)
            decision.destination = _unique_destination(folder, decision.source.name, used_destinations)
            used_destinations.add(str(decision.destination).casefold())
    return ordered


def _organize_plan(
    root: Path,
    inputs: _OrganizationInputs,
    snapshot: dict,
    decisions: Sequence[_OrganizationDecision],
    embedder: Embedder,
    settings: Settings,
) -> dict:
    return {
        "type": "smart-sort-plan",
        "version": 1,
        "root": str(root),
        "source": (
            inputs.source.relative_to(root).as_posix()
            if settings.source is not None else None
        ),
        "recursive": settings.recursive,
        "excludes": list(settings.excludes),
        "backend": embedder.name,
        "thresholds": {
            "match": settings.match_threshold,
            "margin": settings.match_margin,
            "new_group": settings.new_group_threshold,
            "min_group_size": settings.min_group_size,
        },
        "inputs": snapshot,
        "decisions": [
            {
                "source": decision.source.relative_to(root).as_posix(),
                "destination": (
                    decision.destination.relative_to(root).as_posix()
                    if decision.destination is not None else None
                ),
                "classification": decision.kind,
                "score": decision.score,
                "reason": decision.reason,
                "approved": False,
            }
            for decision in decisions
        ],
    }


def _print_organization(root: Path, decisions: Sequence[_OrganizationDecision]) -> None:
    counts = Counter(decision.kind for decision in decisions)
    print(
        f"{len(decisions)} incoming files: {counts['existing']} existing-folder suggestions, "
        f"{counts['new']} new-folder suggestions, {counts['review']} need review"
    )
    for decision in decisions:
        source = decision.source.relative_to(root).as_posix()
        destination = (
            decision.destination.relative_to(root).as_posix()
            if decision.destination is not None else "(needs review)"
        )
        score = f", score {decision.score:.3f}" if decision.score is not None else ""
        print(f"  [{decision.kind}] {source} -> {destination}{score}: {decision.reason}")


def _windows_reserved_component(name: str) -> bool:
    stem = name.split(".", 1)[0].rstrip(" ").upper()
    reserved = {"CON", "PRN", "AUX", "NUL"}
    reserved.update(f"COM{number}" for number in range(1, 10))
    reserved.update(f"LPT{number}" for number in range(1, 10))
    return (
        stem in reserved
        or name.rstrip(" .") != name
        or re.search(r'[<>:"/\\|?*\x00-\x1f]', name) is not None
    )


def run_organize(settings: Settings, root: Path, embedder: Embedder | None = None) -> int:
    if settings.apply:
        LOG.error("Organizing suggestions require a saved, manually approved plan.")
        return 1
    watch = Stopwatch()
    plan_path = (
        Path(os.path.abspath(os.fspath(settings.save_plan.expanduser())))
        if settings.save_plan is not None else None
    )
    try:
        if (settings.match_threshold is None) != (settings.match_margin is None):
            raise ValueError("Match threshold and margin must be supplied together")
        for value, name, upper in (
            (settings.match_threshold, "match threshold", 1.0),
            (settings.match_margin, "match margin", 2.0),
            (settings.new_group_threshold, "new-group threshold", 1.0),
        ):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= upper):
                raise ValueError(f"{name} must be a finite number between 0 and {upper:g}")
        if settings.min_group_size < 2:
            raise ValueError("Minimum group size must be at least 2")
        with watch.phase("collect"):
            inputs = _organization_inputs(
                root, settings.source, settings.recursive, settings.excludes, plan_path
            )
            if not inputs.incoming:
                LOG.error("No eligible incoming files found in %s", inputs.source)
                return 1
        with watch.phase("model"):
            embedder = embedder or build_embedder(settings)
        media_kinds = tuple(
            kind for kind in settings.media
            if kind in getattr(embedder, "media_kinds", frozenset())
        )
        with watch.phase("fingerprint"):
            inputs = _organization_inputs(
                root, settings.source, settings.recursive, settings.excludes, plan_path
            )
            if not inputs.incoming:
                raise OSError(f"No eligible incoming files remain in {inputs.source}")
            before = _organization_snapshot(root, inputs, set())
            files = inputs.incoming + [
                path for paths in inputs.references.values() for path in paths
            ]
            fingerprints, fingerprint_skips = _organization_hashes(root, files, media_kinds)
            failed_paths = {skip.path for skip in fingerprint_skips}
        with watch.phase("extract"):
            documents, skipped = load_documents(
                [path for path in files if path not in failed_paths],
                settings.max_chars, settings.max_chunks, settings.max_pdf_pages,
                settings.min_chars, settings.jobs, media_kinds,
            )
            skipped.extend(fingerprint_skips)
        hash_paths = set(fingerprints)
        with watch.phase("embed"):
            matrix, documents, embed_skips = embed_documents(
                documents, embedder, root / STATE_DIR if settings.use_cache else None,
                settings.use_cache,
            )
            skipped.extend(embed_skips)
        with watch.phase("verify"):
            current = _organization_inputs(
                root, settings.source, settings.recursive, settings.excludes, plan_path
            )
            snapshot = _organization_snapshot(root, current, hash_paths)
            if _snapshot_metadata(before) != _snapshot_metadata(snapshot):
                raise OSError("Inputs changed during preview; run it again")
            expected_hashes = {
                path.relative_to(root).as_posix(): digest for path, digest in fingerprints.items()
            }
            if _snapshot_hashes(snapshot) != expected_hashes:
                raise OSError("Input content changed during preview; run it again")
        with watch.phase("cluster"):
            decisions = _organization_decisions(
                root, current, documents, matrix, skipped, embedder, settings
            )
        plan = _organize_plan(root, current, snapshot, decisions, embedder, settings)
        if plan_path is not None:
            if plan_path.is_relative_to(root):
                assert_safe_path(root, plan_path, allow_missing=True)
            with plan_path.open("x", encoding="utf-8") as handle:
                json.dump(plan, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
    except (OSError, ValueError, RuntimeError, EmbedderError) as exc:
        LOG.error("Could not prepare organizing preview: %s", exc)
        return 1

    _print_organization(root, decisions)
    if plan_path is not None:
        print(f"\nPlan saved: {plan_path}")
        print("Review destinations and set approved to true for each file to move.")
    if settings.timings:
        watch.print_summary()
    print("Preview only: no files were moved; Smart-Sort has not benchmarked these thresholds.")
    return 0


def apply_organize_plan(root: Path, plan_path: Path, apply: bool = False) -> int:
    try:
        root = resolve_root(root)
        plan_path = Path(os.path.abspath(os.fspath(plan_path.expanduser())))
        if is_link_or_reparse(plan_path):
            raise ValueError(f"Plan is missing or is a link: {plan_path}")
        if plan_path.is_relative_to(root):
            assert_safe_path(root, plan_path)
        with plan_path.open("rb") as handle:
            plan_bytes = handle.read()
        plan_digest = hashlib.sha256(plan_bytes).hexdigest()
        plan = json.loads(plan_bytes)
        if not isinstance(plan, dict) or plan.get("type") != "smart-sort-plan" or plan.get("version") != 1:
            raise ValueError("Not a supported Smart-Sort organizing plan")
        plan_root = plan.get("root")
        if not isinstance(plan_root, str) or not Path(plan_root).is_absolute():
            raise ValueError("Plan root must be an absolute path")
        if not os.path.samefile(root, resolve_root(Path(plan_root))):
            raise ValueError("Plan belongs to a different root")
        source_value = plan.get("source")
        if source_value is None:
            source = None
        else:
            source = safe_relative_path(root, source_value)
        recursive = plan.get("recursive")
        excludes = plan.get("excludes")
        snapshot = plan.get("inputs")
        decisions = plan.get("decisions")
        if (
            not isinstance(recursive, bool)
            or not isinstance(excludes, list)
            or not all(isinstance(pattern, str) for pattern in excludes)
            or not isinstance(snapshot, dict)
            or not isinstance(decisions, list)
        ):
            raise ValueError("Plan has invalid discovery or decision data")
        excludes = list(DEFAULT_EXCLUDES) + excludes
        incoming_records = snapshot.get("incoming")
        categories = snapshot.get("categories")
        if not isinstance(incoming_records, list) or not isinstance(categories, list):
            raise ValueError("Plan has invalid input snapshots")
        records = list(incoming_records)
        for category in categories:
            if not isinstance(category, dict) or not isinstance(category.get("files"), list):
                raise ValueError("Plan has invalid category snapshots")
            records.extend(category["files"])
        hash_paths: set[Path] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Plan has an invalid input record")
            path = safe_relative_path(root, record.get("path"))
            digest = record.get("sha256")
            if digest is not None:
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Plan has an invalid input checksum")
                hash_paths.add(path)

        def verify_inputs() -> None:
            if is_link_or_reparse(plan_path) or sha256_file(plan_path) != plan_digest:
                raise ValueError("Plan changed while being applied; review it again")
            if plan_path.is_relative_to(root):
                assert_safe_path(root, plan_path)
            current = _organization_inputs(root, source, recursive, excludes, plan_path)
            actual = _organization_snapshot(root, current, hash_paths)
            if actual != snapshot:
                raise ValueError("Inputs or category profiles changed; create a new preview")

        verify_inputs()
        incoming_by_path = {
            record["path"]: record for record in incoming_records if isinstance(record, dict)
        }
        moves: list[Move] = []
        used_sources: set[str] = set()
        used_destinations: set[str] = set()
        for decision in decisions:
            if not isinstance(decision, dict) or not isinstance(decision.get("approved"), bool):
                raise ValueError("Every decision must have a boolean approved field")
            source_path = safe_relative_path(root, decision.get("source"))
            source_key = source_path.relative_to(root).as_posix()
            if source_key in used_sources or source_key not in incoming_by_path:
                raise ValueError(f"Plan has a duplicate or unknown source: {source_key}")
            used_sources.add(source_key)
            if not decision["approved"]:
                continue
            record = incoming_by_path[source_key]
            digest, size = record.get("sha256"), record.get("size")
            if not isinstance(digest, str) or not isinstance(size, int):
                raise ValueError(f"Approved file has no usable fingerprint: {source_key}")
            destination = safe_relative_path(root, decision.get("destination"))
            relative = destination.relative_to(root)
            if (
                len(relative.parts) < 2
                or (source is not None and relative.parts[0].casefold() == source.relative_to(root).parts[0].casefold())
                or (os.name == "nt" and any(_windows_reserved_component(part) for part in relative.parts))
            ):
                raise ValueError(f"Destination is not an eligible category: {destination}")
            parent = root
            for part in relative.parts[:-1]:
                parent = parent / part
                if (
                    part.startswith(".")
                    or _matches_any(part, excludes)
                    or (parent.exists() and is_hidden_or_system(parent))
                ):
                    raise ValueError(f"Destination contains an excluded or hidden folder: {parent}")
            if destination.name.startswith("~$") or (
                os.name != "nt" and destination.name.startswith(".")
            ):
                raise ValueError(f"Destination is a hidden or lock file: {destination}")
            destination_key = str(destination).casefold()
            if destination_key in used_destinations or os.path.lexists(destination):
                raise ValueError(f"Destination already exists or is duplicated: {destination}")
            used_destinations.add(destination_key)
            moves.append(Move(source_path, destination, 0, digest, size))

        print(f"{len(moves)} approved moves")
        for move in moves:
            print(f"  {move.src.relative_to(root).as_posix()} -> {move.dst.relative_to(root).as_posix()}")
        if not apply:
            print("Dry run: no files were moved. Add --apply to execute the reviewed plan.")
            return 0
        if not moves:
            raise ValueError("No files were approved in the plan")
        result = apply_moves(
            root, moves,
            {"backend": plan.get("backend"), "plan": str(plan_path), "files": len(moves)},
            preflight=verify_inputs,
        )
    except (OSError, ValueError, TypeError) as exc:
        LOG.error("Could not apply organizing plan: %s", exc)
        return 1

    print(f"Moved {result.moved} files.")
    if result.failed:
        print(f"Failed: {result.failed} files; inspect the manifest for recovery.")
    if result.manifest is not None:
        print(f"Manifest: {result.manifest}")
        print(f"Undo or recover with: {_undo_command(result.manifest)}")
    return 0 if result.failed == 0 else 1


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


def _undo_command(manifest: Path) -> str:
    arguments = [sys.executable, str(Path(__file__).resolve()), "--undo", str(manifest), "--apply"]
    if os.name == "nt":
        quoted = ["'" + argument.replace("'", "''") + "'" for argument in arguments]
        return "& " + " ".join(quoted)
    return shlex.join(arguments)


def run(settings: Settings, embedder: Embedder | None = None) -> int:
    try:
        root = resolve_root(settings.root)
    except (OSError, ValueError) as exc:
        LOG.error("Unsafe or inaccessible root: %s", exc)
        return 1
    if settings.organize:
        return run_organize(settings, root, embedder)
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
    except (OSError, RuntimeError, ValueError) as exc:
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
        try:
            moves = plan_moves(root, documents, labels, names)
        except (OSError, ValueError) as exc:
            LOG.error("Cannot plan moves safely: %s", exc)
            return 1

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

    try:
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
    except (OSError, ValueError) as exc:
        LOG.error("Could not start the applied run: %s", exc)
        return 1
    if result.manifest is None:
        LOG.error("Could not create an undo manifest (%d failed).", result.failed)
        return 1

    folder_word = "folder" if len(names) == 1 else "folders"
    print(f"\nMoved {result.moved} files into {len(names)} {folder_word}.")
    if result.failed:
        print(f"Failed: {result.failed} files; inspect the manifest for recovery.")
    print(f"Manifest: {result.manifest}")
    print(f"Undo or recover with: {_undo_command(result.manifest)}")
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
            '  smart_sort.py "D:\\Docs" --organize --save-plan "D:\\Plans\\sort.json"\n'
            '  smart_sort.py "D:\\Docs" --apply-plan "D:\\Plans\\sort.json" --apply\n'
            '  smart_sort.py --undo ".smart_sort/manifest-20250101-120000.jsonl" --apply\n'
        ),
    )
    parser.add_argument("path", nargs="?", help="Folder to organize.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag the run is a dry run.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--undo",
        metavar="MANIFEST",
        help="Reverse a previous run using its JSONL manifest (dry run unless --apply).",
    )
    modes.add_argument(
        "--organize",
        action="store_true",
        help="Preview incoming files against existing folders and possible new groups.",
    )
    modes.add_argument(
        "--apply-plan",
        metavar="PLAN",
        help="Preview or apply manually approved decisions from an organizing plan.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Incoming folder inside the root for --organize (default: loose root files).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        help="Explicit minimum cosine similarity for existing-folder suggestions.",
    )
    parser.add_argument(
        "--match-margin",
        type=float,
        help="Explicit minimum score gap between the best two existing folders.",
    )
    parser.add_argument(
        "--new-group-threshold",
        type=float,
        help="Explicit minimum pairwise cosine similarity for new-folder suggestions.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum files in a proposed new group (default: 2).",
    )
    parser.add_argument(
        "--save-plan",
        type=Path,
        help="Save an organizing preview as a new JSON plan for manual review.",
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

    if not args.organize and (
        args.source is not None
        or args.match_threshold is not None
        or args.match_margin is not None
        or args.new_group_threshold is not None
        or args.min_group_size != 2
        or args.save_plan is not None
    ):
        parser.error("organizing thresholds, --source, and --save-plan require --organize")
    if args.organize:
        if args.apply:
            parser.error("--organize is preview-only; review a saved plan and use --apply-plan")
        if args.k is not None or args.auto_k:
            parser.error("--k and --auto-k apply to clustering, not --organize")
        if args.recursive and args.source is None:
            parser.error("--organize --recursive requires --source")
        if (args.match_threshold is None) != (args.match_margin is None):
            parser.error("--match-threshold and --match-margin must be supplied together")
        for value, name, upper in (
            (args.match_threshold, "--match-threshold", 1.0),
            (args.match_margin, "--match-margin", 2.0),
            (args.new_group_threshold, "--new-group-threshold", 1.0),
        ):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= upper):
                parser.error(f"{name} must be a finite number between 0 and {upper:g}")
        if args.min_group_size < 2:
            parser.error("--min-group-size must be at least 2")

    if args.undo:
        manifest = Path(args.undo).expanduser()
        if not manifest.is_file():
            parser.error(f"manifest not found: {manifest}")
        started = time.perf_counter()
        try:
            restored, failed = undo_manifest(manifest, dry_run=not args.apply)
        except KeyboardInterrupt:
            LOG.warning("Undo interrupted; use the same manifest to resume recovery.")
            return 130
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

    if args.apply_plan:
        started = time.perf_counter()
        try:
            result = apply_organize_plan(root, Path(args.apply_plan), apply=args.apply)
        except KeyboardInterrupt:
            LOG.warning("Interrupted; use the manifest under .smart_sort to recover the run.")
            return 130
        if args.timings:
            print(f"\nTimings:\n  {'plan':<12} {time.perf_counter() - started:8.3f} s")
        return result

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
        organize=args.organize,
        source=args.source,
        match_threshold=args.match_threshold,
        match_margin=args.match_margin,
        new_group_threshold=args.new_group_threshold,
        min_group_size=args.min_group_size,
        save_plan=args.save_plan,
    )

    try:
        return run(settings)
    except KeyboardInterrupt:
        LOG.warning("Interrupted; use the manifest under .smart_sort to recover the run.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
