"""Tests for smart_sort.py, runnable with `python -m unittest discover -s tests`."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import smart_sort as ss  # noqa: E402


class KeywordEmbedder:
    """Deterministic stand-in for a real embedding model."""

    name = "keyword-test"
    cacheable = False

    def embed_corpus(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(6, dtype=np.float32)
            if "invoice" in lowered or "payment" in lowered:
                vector[0] = 1.0
            if "poem" in lowered or "verse" in lowered:
                vector[1] = 1.0
            if "recipe" in lowered or "flour" in lowered:
                vector[2] = 1.0
            vectors.append(vector)
        return ss._l2_normalize(np.asarray(vectors, dtype=np.float32))


class CountingEmbedder:
    """Cacheable embedder that records how often it is invoked."""

    name = "counting"
    cacheable = True

    def __init__(self):
        self.calls = 0
        self.batch_sizes = []

    def embed_corpus(self, texts):
        self.calls += 1
        self.batch_sizes.append(len(texts))
        vectors = [[1.0, float(len(text) % 5) + 1.0] for text in texts]
        return ss._l2_normalize(np.asarray(vectors, dtype=np.float32))


class OtherEmbedder(CountingEmbedder):
    name = "other"


class SmartSortTestCase(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self):
        self._temp.cleanup()

    def write(self, relative: str, text: str, encoding: str = "utf-8") -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding=encoding)
        return path

    def test_collect_files_skips_hidden_state_and_cluster_dirs(self):
        self.write("top.txt", "hello world")
        self.write(".hidden.txt", "hidden")
        self.write("Cluster_1/sorted.txt", "already sorted")
        self.write(".smart_sort/cache.npz", "state")
        self.write("sub/nested.txt", "nested content")
        self.write("server.log", "log line")

        top_level = ss.collect_files(self.root)
        self.assertEqual([p.name for p in top_level], ["server.log", "top.txt"])

        filtered = ss.collect_files(self.root, excludes=ss.DEFAULT_EXCLUDES + ("*.log",))
        self.assertEqual([p.name for p in filtered], ["top.txt"])

        recursive = ss.collect_files(self.root, recursive=True)
        self.assertEqual([p.name for p in recursive], ["server.log", "top.txt", "nested.txt"])

    def test_split_chunks_samples_first_and_last_content(self):
        text = "A" * 1000 + "B" * 1000
        chunks = ss.split_chunks(text, max_chars=100, max_chunks=4)

        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertTrue(chunks[0].startswith("A"))
        self.assertTrue(chunks[-1].endswith("B"))

    def test_split_chunks_covers_short_text_fully(self):
        text = "".join(str(i % 10) for i in range(350))
        chunks = ss.split_chunks(text, max_chars=100, max_chunks=4)

        self.assertEqual(len(chunks), 4)
        self.assertEqual("".join(chunks), text)

    def test_read_text_falls_back_to_cp1252(self):
        path = self.root / "latin.txt"
        path.write_bytes("café notes with enough text".encode("cp1252"))

        self.assertEqual(ss.extract_text(path), "café notes with enough text")

    def test_binary_file_is_rejected(self):
        path = self.root / "data.bin"
        path.write_bytes(b"\x00\x01\x02binary")

        with self.assertRaises(ss.ExtractionError):
            ss.extract_text(path)

    def test_docx_text_is_extracted(self):
        import docx

        document = docx.Document()
        document.add_paragraph("Quarterly revenue grew twelve percent and churn fell.")
        path = self.root / "report.docx"
        document.save(str(path))

        self.assertIn("revenue grew", ss.extract_text(path))

    def make_docx(self, name: str, text: str) -> Path:
        import docx

        document = docx.Document()
        document.add_paragraph(text)
        path = self.root / name
        document.save(str(path))
        return path

    def test_auto_worker_count_heuristic(self):
        small = [self.write(f"small-{index}.docx", "x") for index in range(2)]
        self.assertEqual(ss._auto_worker_count(small), 1)
        self.assertEqual(ss._auto_worker_count(small[:1]), 1)

        big = self.root / "big.pdf"
        with big.open("wb") as handle:
            handle.truncate(9 * 1024 * 1024)
        self.assertGreater(ss._auto_worker_count([big, small[0]]), 1)

    def test_parallel_extraction_matches_sequential(self):
        for index in range(4):
            self.make_docx(
                f"report-{index}.docx",
                f"Report section {index} about revenue, churn and the hiring plan.",
            )
        self.write("notes.txt", "plain notes about the roadmap and the budget")

        files = sorted(path for path in self.root.iterdir() if path.is_file())
        sequential, skipped_sequential = ss.load_documents(files, 4000, 2, 5, 5, jobs=1)
        parallel, skipped_parallel = ss.load_documents(files, 4000, 2, 5, 5, jobs=2)

        self.assertEqual(skipped_sequential, [])
        self.assertEqual(skipped_parallel, [])
        self.assertEqual([doc.path for doc in sequential], [doc.path for doc in parallel])
        self.assertEqual([doc.text for doc in sequential], [doc.text for doc in parallel])

    def test_parallel_extraction_failure_falls_back(self):
        for index in range(2):
            self.make_docx(f"report-{index}.docx", f"Report {index} about revenue and churn.")

        files = sorted(path for path in self.root.iterdir() if path.is_file())
        original = ss._parallel_heavy_results

        def explode(*args, **kwargs):
            raise RuntimeError("pool unavailable")

        ss._parallel_heavy_results = explode
        try:
            documents, skipped = ss.load_documents(files, 4000, 2, 5, 5, jobs=4)
        finally:
            ss._parallel_heavy_results = original

        self.assertEqual(len(documents), 2)
        self.assertEqual(skipped, [])

    def test_plan_moves_disambiguates_same_filename(self):
        first = self.write("a/report.txt", "invoice payment")
        second = self.write("b/report.txt", "invoice payment")
        documents = [
            ss.Document(first, "invoice payment", ["invoice payment"]),
            ss.Document(second, "invoice payment", ["invoice payment"]),
        ]
        labels = np.array([0, 0])
        moves = ss.plan_moves(self.root, documents, labels, {0: "Cluster_1_invoices"})

        destinations = [move.dst for move in moves]
        self.assertEqual(len(set(destinations)), 2)
        self.assertTrue(all(dst.parent.name == "Cluster_1_invoices" for dst in destinations))
        self.assertEqual(sorted(dst.name for dst in destinations), ["report (2).txt", "report.txt"])

    def test_build_cluster_names_are_unique(self):
        documents = [
            ss.Document(self.root / "a.txt", "shared text", ["shared text"]),
            ss.Document(self.root / "b.txt", "shared text", ["shared text"]),
        ]
        labels = np.array([0, 1])
        names = ss.build_cluster_names(documents, labels)

        self.assertEqual(len(set(names.values())), 2)
        self.assertTrue(all(name.startswith("Cluster_") for name in names.values()))

    def test_choose_k_clamps_explicit_request(self):
        self.assertEqual(ss.choose_k(3, requested=10), 3)
        self.assertEqual(ss.choose_k(200), 10)
        self.assertEqual(ss.choose_k(2), 2)

    def test_embedding_cache_dedupes_and_skips_reembedding(self):
        documents = [
            ss.Document(self.root / "a.txt", "alpha text", ["alpha text"]),
            ss.Document(self.root / "b.txt", "alpha text", ["alpha text"]),
        ]
        cache_dir = self.root / ss.STATE_DIR

        first = CountingEmbedder()
        first_matrix = ss.embed_documents(documents, first, cache_dir, use_cache=True)
        self.assertEqual(first.calls, 1)
        self.assertEqual(first.batch_sizes, [1])  # identical chunks embedded once
        self.assertEqual(len(list(cache_dir.glob("embeddings-*.npz"))), 1)

        second = CountingEmbedder()
        second_matrix = ss.embed_documents(documents, second, cache_dir, use_cache=True)
        self.assertEqual(second.calls, 0)
        np.testing.assert_allclose(first_matrix, second_matrix)

        other = OtherEmbedder()
        ss.embed_documents(documents, other, cache_dir, use_cache=True)
        self.assertEqual(other.calls, 1)  # cache identity differs, so re-embed

    def test_end_to_end_apply_and_undo(self):
        for index in range(3):
            self.write(f"invoice-{index}.txt", f"invoice payment number {index} due today")
            self.write(f"poem-{index}.txt", f"poem verse about the sea number {index}")
        (self.root / "payload.bin").write_bytes(b"\x00\x01\x02binary")

        settings = ss.Settings(root=self.root, k=2, apply=True, backend="tfidf")
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = ss.run(settings, embedder=KeywordEmbedder())

        self.assertEqual(exit_code, 0)
        self.assertIn("Moved 6 files", output.getvalue())

        cluster_dirs = sorted(
            p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("Cluster_")
        )
        self.assertEqual(len(cluster_dirs), 2)
        for directory in cluster_dirs:
            self.assertEqual(len(list(directory.iterdir())), 3)

        invoice_parents = {
            path.relative_to(self.root).parts[0] for path in self.root.rglob("invoice-*.txt")
        }
        poem_parents = {
            path.relative_to(self.root).parts[0] for path in self.root.rglob("poem-*.txt")
        }
        self.assertEqual(len(invoice_parents), 1)
        self.assertEqual(len(poem_parents), 1)
        self.assertNotEqual(invoice_parents, poem_parents)

        manifests = sorted((self.root / ss.STATE_DIR).glob("manifest-*.jsonl"))
        self.assertEqual(len(manifests), 1)
        records = [
            json.loads(line) for line in manifests[0].read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(records[0]["type"], "header")
        self.assertEqual(sum(1 for record in records if record["type"] == "move"), 6)

        dry_restored, dry_failed = ss.undo_manifest(manifests[0], dry_run=True)
        self.assertEqual((dry_restored, dry_failed), (6, 0))
        self.assertEqual(len(list(self.root.rglob("invoice-*.txt"))), 3)

        restored, failed = ss.undo_manifest(manifests[0], dry_run=False)
        self.assertEqual((restored, failed), (6, 0))
        for index in range(3):
            self.assertTrue((self.root / f"invoice-{index}.txt").is_file())
            self.assertTrue((self.root / f"poem-{index}.txt").is_file())
        self.assertEqual(
            [p for p in self.root.iterdir() if p.is_dir() and p.name.startswith("Cluster_")],
            [],
        )

    def test_tfidf_backend_dry_run_moves_nothing(self):
        for index in range(3):
            self.write(f"invoice-{index}.txt", f"invoice payment ledger entry {index}")
            self.write(f"poem-{index}.txt", f"poem verse rhythm stanza {index}")

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = ss.main([str(self.root), "--backend", "tfidf", "--k", "2"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Dry run", output.getvalue())
        self.assertEqual(len(list(self.root.glob("*.txt"))), 6)
        self.assertFalse((self.root / ss.STATE_DIR).exists())

    def test_timings_flag_prints_phases(self):
        for index in range(4):
            self.write(f"doc-{index}.txt", f"document number {index} about invoices and payment")

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = ss.main([str(self.root), "--backend", "tfidf", "--timings"])

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Timings:", rendered)
        for phase in ("collect", "extract", "embed", "cluster", "wall clock"):
            self.assertIn(phase, rendered)


if __name__ == "__main__":
    unittest.main()
