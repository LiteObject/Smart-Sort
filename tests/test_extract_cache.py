"""Unicode, bounded reads, and model identity tests."""

from __future__ import annotations

import codecs
import os
import sys
import tempfile
import tracemalloc
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import smart_sort as ss  # noqa: E402
import text_extract as te  # noqa: E402


class ExtractAndCacheTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_unicode_text_is_not_mistaken_for_binary(self):
        samples = {
            "chinese": "\u8fd9\u662f\u4e00\u4efd\u5173\u4e8e\u5b63\u5ea6\u6536\u5165\u7684\u6587\u4ef6\u3002",
            "russian": "\u041f\u0440\u0438\u0432\u0435\u0442, \u044d\u0442\u043e \u0442\u0435\u043a\u0441\u0442\u043e\u0432\u044b\u0439 \u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442.",
        }
        for language, text in samples.items():
            for encoding in ("utf-8", "utf-16", "utf-32"):
                with self.subTest(language=language, encoding=encoding):
                    path = self.root / f"{language}-{encoding}.txt"
                    path.write_bytes((text * 3).encode(encoding))
                    self.assertEqual(te.extract_text(path), text * 3)

        text = samples["chinese"] * 3
        path = self.root / "big-endian.unknown"
        path.write_bytes(codecs.BOM_UTF16_BE + text.encode("utf-16-be"))
        self.assertEqual(te.extract_text(path), text)

    def test_text_sample_can_end_midway_through_a_utf8_character(self):
        text = "a" * 8191 + "\u4e2d" + "more text"
        path = self.root / "boundary.txt"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(te.extract_text(path), text)

    def test_large_unsupported_files_are_sniffed_with_bounded_memory(self):
        for suffix in (".iso", ".txt"):
            with self.subTest(suffix=suffix):
                path = self.root / f"large{suffix}"
                with path.open("wb") as handle:
                    handle.truncate(16 * 1024 * 1024)
                tracemalloc.start()
                try:
                    with self.assertRaises(te.ExtractionError):
                        te.extract_text(path)
                    _current, peak = tracemalloc.get_traced_memory()
                finally:
                    tracemalloc.stop()
                self.assertLess(peak, 2 * 1024 * 1024)

    def test_image_only_pdf_is_reported_explicitly(self):
        path = self.root / "scanned.pdf"
        path.write_bytes(b"placeholder")
        with mock.patch.object(te, "_extract_pdf", return_value=""):
            documents, skipped = ss.load_documents([path], 4000, 4, 20, 20, jobs=1)
        self.assertEqual(documents, [])
        self.assertEqual(len(skipped), 1)
        self.assertIn("possibly a scanned PDF", skipped[0].reason)

    def test_gguf_cache_identity_tracks_model_content(self):
        class FakeLlama:
            def __init__(self, model_path, **_kwargs):
                self.path = Path(model_path)
                self.calls = 0

            def create_embedding(self, _text):
                self.calls += 1
                vector = [1.0, 0.0] if self.path.read_bytes() == b"AAAA" else [0.0, 1.0]
                return {"data": [{"embedding": vector}]}

        module = types.ModuleType("llama_cpp")
        module.Llama = FakeLlama
        first_path = self.root / "one" / "same.gguf"
        second_path = self.root / "two" / "same.gguf"
        first_path.parent.mkdir()
        second_path.parent.mkdir()
        first_path.write_bytes(b"AAAA")
        second_path.write_bytes(b"BBBB")
        document = ss.Document(self.root / "sample.txt", "document", ["document"])
        cache = self.root / ss.STATE_DIR

        with mock.patch.dict(sys.modules, {"llama_cpp": module}):
            first = ss.LlamaEmbedder(first_path)
            first_matrix, _, _ = ss.embed_documents([document], first, cache)
            second = ss.LlamaEmbedder(second_path)
            second_matrix, _, _ = ss.embed_documents([document], second, cache)
            self.assertNotEqual(first.name, second.name)
            self.assertEqual(second._model.calls, 1)
            self.assertFalse(np.array_equal(first_matrix, second_matrix))

            original = first_path.stat()
            first_path.write_bytes(b"CCCC")
            os.utime(first_path, ns=(original.st_atime_ns, original.st_mtime_ns))
            replaced = ss.LlamaEmbedder(first_path)
            ss.embed_documents([document], replaced, cache)
            self.assertNotEqual(first.name, replaced.name)
            self.assertEqual(replaced._model.calls, 1)

    def test_invalid_embeddings_are_skipped_and_not_cached(self):
        class PartialEmbedder:
            name = "partial-embedding-test"
            cacheable = True
            media_kinds = frozenset()

            def __init__(self):
                self.batch_sizes = []

            def embed_corpus(self, texts):
                self.batch_sizes.append(len(texts))
                values = {
                    "valid": [1.0, 0.0],
                    "zero": [0.0, 0.0],
                    "nan": [float("nan"), 1.0],
                }
                return np.asarray([values[text] for text in texts], dtype=np.float32)

        documents = [
            ss.Document(self.root / "good.txt", "valid", ["valid"]),
            ss.Document(self.root / "zero.txt", "zero", ["zero"]),
            ss.Document(self.root / "nan.txt", "nan", ["nan"]),
            ss.Document(self.root / "mixed.txt", "valid zero", ["valid", "zero"]),
        ]
        cache = self.root / ss.STATE_DIR
        first = PartialEmbedder()
        matrix, kept, skipped = ss.embed_documents(documents, first, cache)
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual([document.path.name for document in kept], ["good.txt"])
        self.assertEqual({skip.path.name for skip in skipped}, {"zero.txt", "nan.txt", "mixed.txt"})
        self.assertTrue(all("invalid" in skip.reason for skip in skipped))
        self.assertEqual(first.batch_sizes, [3])

        second = PartialEmbedder()
        matrix, kept, skipped = ss.embed_documents(documents, second, cache)
        self.assertEqual(second.batch_sizes, [2])
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(skipped), 3)

    def test_legacy_clustering_skips_an_invalid_embedding(self):
        class PartialEmbedder:
            name = "legacy-partial-test"
            cacheable = False
            media_kinds = frozenset()

            def embed_corpus(self, texts):
                return np.asarray(
                    [[1.0, 0.0] if text == "valid" else [0.0, 0.0] for text in texts],
                    dtype=np.float32,
                )

        for name, text in (("a.txt", "valid"), ("b.txt", "valid"), ("bad.txt", "zero")):
            (self.root / name).write_text(text, encoding="utf-8")
        settings = ss.Settings(root=self.root, k=1, min_chars=1, use_cache=False)
        with self.assertLogs("smart-sort", level="WARNING") as logs:
            self.assertEqual(ss.run(settings, embedder=PartialEmbedder()), 0)
        self.assertIn("invalid text chunk embedding", "\n".join(logs.output))
        self.assertTrue((self.root / "bad.txt").exists())

    def test_media_cache_identity_tracks_content_not_only_metadata(self):
        class MediaEmbedder:
            name = "media-content-test"
            cacheable = True
            media_kinds = frozenset({"image"})

            def __init__(self):
                self.calls = 0

            def embed_corpus(self, texts):
                return np.empty((0, 2), dtype=np.float32)

            def embed_media(self, items):
                self.calls += 1
                return np.asarray(
                    [[1.0, 0.0] if path.read_bytes() == b"AAAA" else [0.0, 1.0]
                     for path, _kind in items],
                    dtype=np.float32,
                )

        path = self.root / "photo.png"
        path.write_bytes(b"AAAA")
        document = ss.Document(path, "", [], "image")
        cache = self.root / ss.STATE_DIR
        first = MediaEmbedder()
        original, _, _ = ss.embed_documents([document], first, cache)
        metadata = path.stat()
        path.write_bytes(b"BBBB")
        os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
        second = MediaEmbedder()
        replaced, _, _ = ss.embed_documents([document], second, cache)
        self.assertEqual(second.calls, 1)
        self.assertFalse(np.array_equal(original, replaced))

    def test_unreadable_media_fingerprint_is_reported_per_file(self):
        class MediaEmbedder:
            name = "unreadable-media-test"
            cacheable = True
            media_kinds = frozenset({"image"})

            def embed_corpus(self, texts):
                return np.empty((0, 2), dtype=np.float32)

            def embed_media(self, items):
                raise AssertionError("Unreadable media must not be sent to the backend")

        path = self.root / "photo.png"
        path.write_bytes(b"AAAA")
        document = ss.Document(path, "", [], "image")
        with mock.patch.object(ss, "_media_cache_key", side_effect=OSError("cannot read")):
            with self.assertLogs("smart-sort", level="WARNING"):
                matrix, kept, skipped = ss.embed_documents(
                    [document], MediaEmbedder(), self.root / ss.STATE_DIR
                )
        self.assertEqual(len(matrix), 0)
        self.assertEqual(kept, [])
        self.assertEqual(skipped[0].reason, "cannot fingerprint image")

    def test_multimodal_cache_requires_immutable_revision_and_safe_auto_dtype(self):
        class FakeCuda:
            def __init__(self):
                self.queries = []

            def is_available(self):
                return True

            def device(self, value):
                self.queries.append(value)
                return nullcontext()

            def is_bf16_supported(self, including_emulation=True):
                self.queries.append(including_emulation)
                return False

        torch = types.ModuleType("torch")
        torch.cuda = FakeCuda()
        torch.bfloat16 = "torch.bfloat16"
        torch.float16 = "torch.float16"
        torch.float32 = "torch.float32"

        class FakeModel:
            revision = "a" * 40
            last_kwargs = None

            def __init__(self, _name, **kwargs):
                FakeModel.last_kwargs = kwargs
                self.prompts = {"Document": "document"}

            def children(self):
                config = types.SimpleNamespace(_commit_hash=FakeModel.revision)
                return [types.SimpleNamespace(auto_model=types.SimpleNamespace(config=config))]

        transformers = types.ModuleType("sentence_transformers")
        transformers.SentenceTransformer = FakeModel
        with mock.patch.dict(
            sys.modules, {"torch": torch, "sentence_transformers": transformers}
        ), mock.patch.object(ss, "_prepare_windows_ffmpeg"):
            first = ss.SentenceTransformerEmbedder("remote/model", device="auto", dtype="auto")
            self.assertEqual(FakeModel.last_kwargs["model_kwargs"]["torch_dtype"], torch.float32)
            self.assertEqual(torch.cuda.queries, ["cuda", False])
            self.assertTrue(first.cacheable)
            self.assertIn(FakeModel.revision, first.name)

            FakeModel.revision = "b" * 40
            second = ss.SentenceTransformerEmbedder("remote/model", device="cpu", dtype="auto")
            self.assertNotEqual(first.name, second.name)
            self.assertTrue(second.cacheable)

            FakeModel.revision = None
            with self.assertLogs("smart-sort", level="WARNING"):
                unpinned = ss.SentenceTransformerEmbedder("remote/model", device="cpu")
            self.assertFalse(unpinned.cacheable)

            local = self.root / "model"
            local.mkdir()
            FakeModel.revision = "a" * 40
            with self.assertLogs("smart-sort", level="WARNING"):
                local_model = ss.SentenceTransformerEmbedder(str(local), device="cpu")
            self.assertFalse(local_model.cacheable)


if __name__ == "__main__":
    unittest.main()
