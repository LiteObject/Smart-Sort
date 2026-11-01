"""Tests for existing-folder suggestions and manually reviewed organizing plans."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

import smart_sort as ss


class TopicEmbedder:
    name = "topic-test"
    cacheable = False
    media_kinds = frozenset()

    def embed_corpus(self, texts):
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("invoice" in lowered),
                    float("poem" in lowered),
                    float("recipe" in lowered),
                    float("travel" in lowered),
                ]
            )
        return ss._l2_normalize(np.asarray(rows, dtype=np.float32))


class OrganizeTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "root"
        self.root.mkdir()
        self.plan = self.base / "plan.json"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def populate(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        self.write("Poems/reference.txt", "poem literature reference document")
        (self.root / "Empty").mkdir()
        self.write("bill.txt", "invoice payment for the current month")
        self.write("recipe-a.txt", "recipe ingredients and baking instructions")
        self.write("recipe-b.txt", "recipe cooking and dinner instructions")
        self.write("ambiguous.txt", "invoice and poem mentioned together")
        self.write("solo.txt", "travel destinations and vacation details")
        (self.root / "binary.bin").write_bytes(b"\x00\x01\x02binary")

    def preview(self, **overrides):
        options = {
            "root": self.root,
            "organize": True,
            "match_threshold": 0.6,
            "match_margin": 0.1,
            "new_group_threshold": 0.9,
            "save_plan": self.plan,
            "use_cache": False,
        }
        options.update(overrides)
        output = io.StringIO()
        with redirect_stdout(output):
            code = ss.run(ss.Settings(**options), embedder=TopicEmbedder())
        self.assertEqual(code, 0)
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        return plan, output.getvalue()

    def approve(self, plan, names):
        for decision in plan["decisions"]:
            decision["approved"] = decision["source"] in names
        self.plan.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def test_preview_matches_existing_groups_novel_and_abstains(self):
        self.populate()
        plan, output = self.preview()
        by_source = {item["source"]: item for item in plan["decisions"]}

        self.assertEqual(by_source["bill.txt"]["classification"], "existing")
        self.assertEqual(by_source["bill.txt"]["destination"], "Invoices/bill.txt")
        self.assertEqual(by_source["recipe-a.txt"]["classification"], "new")
        self.assertEqual(by_source["recipe-b.txt"]["classification"], "new")
        self.assertEqual(
            Path(by_source["recipe-a.txt"]["destination"]).parent,
            Path(by_source["recipe-b.txt"]["destination"]).parent,
        )
        for name in ("ambiguous.txt", "solo.txt", "binary.bin"):
            self.assertEqual(by_source[name]["classification"], "review")
            self.assertIsNone(by_source[name]["destination"])
        self.assertTrue(all(not item["approved"] for item in plan["decisions"]))
        self.assertIn("existing-folder suggestions", output)
        self.assertIn("need review", output)
        self.assertTrue((self.root / "bill.txt").exists())
        self.assertTrue((self.root / "recipe-a.txt").exists())
        self.assertFalse((self.root / by_source["recipe-a.txt"]["destination"]).exists())
        self.assertTrue((self.root / "Invoices" / "reference.txt").exists())

    def test_reviewed_moves_apply_and_undo_without_moving_references(self):
        self.populate()
        plan, _ = self.preview()
        self.approve(plan, {"bill.txt", "recipe-a.txt", "recipe-b.txt"})

        with redirect_stdout(io.StringIO()):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan), 0)
        self.assertTrue((self.root / "bill.txt").exists())

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 0)
        self.assertTrue((self.root / "Invoices" / "bill.txt").exists())
        self.assertTrue((self.root / "Invoices" / "reference.txt").exists())
        self.assertTrue((self.root / "ambiguous.txt").exists())
        self.assertTrue((self.root / "solo.txt").exists())
        manifest = next(
            line.removeprefix("Manifest: ")
            for line in output.getvalue().splitlines()
            if line.startswith("Manifest: ")
        )
        self.assertEqual(ss.undo_manifest(Path(manifest), dry_run=False), (3, 0))
        for name in ("bill.txt", "recipe-a.txt", "recipe-b.txt"):
            self.assertTrue((self.root / name).exists())
        self.assertTrue((self.root / "Invoices" / "reference.txt").exists())

    def test_unapproved_or_ambiguous_decisions_are_not_moved(self):
        self.populate()
        plan, _ = self.preview()
        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue(all(not item["approved"] for item in plan["decisions"]))
        self.assertTrue((self.root / "bill.txt").exists())
        self.assertFalse((self.root / ss.STATE_DIR).exists())

    def test_changed_incoming_content_requires_a_new_preview(self):
        self.populate()
        plan, _ = self.preview()
        self.approve(plan, {"bill.txt"})
        path = self.root / "bill.txt"
        before = path.stat()
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("invoice", "INVOICE"), encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue(path.exists())
        self.assertFalse((self.root / "Invoices" / "bill.txt").exists())

    def test_changed_reference_content_requires_a_new_preview(self):
        self.populate()
        plan, _ = self.preview()
        self.approve(plan, {"bill.txt"})
        path = self.root / "Invoices" / "reference.txt"
        before = path.stat()
        original = path.read_text(encoding="utf-8")
        path.write_text(original.replace("invoice", "INVOICE"), encoding="utf-8")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue((self.root / "bill.txt").exists())
        self.assertFalse((self.root / "Invoices" / "bill.txt").exists())

    def test_content_changed_during_preview_requires_a_new_preview(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        source = self.write("bill.txt", "invoice payment for the current month")
        original_info = source.stat()
        original_embed = ss.embed_documents

        def mutate_after_embedding(*args, **kwargs):
            result = original_embed(*args, **kwargs)
            text = source.read_text(encoding="utf-8").replace("invoice", "poemxyz")
            source.write_text(text, encoding="utf-8")
            os.utime(source, ns=(original_info.st_atime_ns, original_info.st_mtime_ns))
            return result

        settings = ss.Settings(
            root=self.root, organize=True, match_threshold=0.6, match_margin=0.1,
            save_plan=self.plan, use_cache=False,
        )
        with mock.patch.object(ss, "embed_documents", side_effect=mutate_after_embedding):
            with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
                self.assertEqual(ss.run(settings, embedder=TopicEmbedder()), 1)
        self.assertFalse(self.plan.exists())
        self.assertTrue(source.exists())

    def test_new_incoming_file_requires_a_new_preview(self):
        self.populate()
        plan, _ = self.preview()
        self.approve(plan, {"bill.txt"})
        self.write("added.txt", "invoice added after the initial preview")

        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue((self.root / "bill.txt").exists())

    def test_new_category_requires_a_new_preview(self):
        self.populate()
        plan, _ = self.preview()
        self.approve(plan, {"bill.txt"})
        (self.root / "New category").mkdir()

        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue((self.root / "bill.txt").exists())

    def test_existing_destination_or_state_destination_is_rejected(self):
        self.populate()
        plan, _ = self.preview()
        decision = next(item for item in plan["decisions"] if item["source"] == "bill.txt")
        decision["approved"] = True
        decision["destination"] = "Invoices/reference.txt"
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertEqual(
            (self.root / "Invoices" / "reference.txt").read_text(encoding="utf-8"),
            "invoice accounting reference document",
        )

        decision["destination"] = ".smart_sort/anything.txt"
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue((self.root / "bill.txt").exists())

    @unittest.skipUnless(os.name == "nt", "Windows destination names")
    def test_reserved_names_and_alternate_streams_are_rejected(self):
        self.populate()
        plan, _ = self.preview()
        decision = next(item for item in plan["decisions"] if item["source"] == "bill.txt")
        decision["approved"] = True
        for destination in ("Invoices/CON.txt", "Invoices/reference.txt:other"):
            with self.subTest(destination=destination):
                decision["destination"] = destination
                self.plan.write_text(json.dumps(plan), encoding="utf-8")
                with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
                    self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
                self.assertTrue((self.root / "bill.txt").exists())
                self.assertEqual(
                    (self.root / "Invoices" / "reference.txt").read_text(encoding="utf-8"),
                    "invoice accounting reference document",
                )

    def test_manually_approved_review_can_select_a_safe_destination(self):
        self.populate()
        plan, _ = self.preview()
        decision = next(item for item in plan["decisions"] if item["source"] == "ambiguous.txt")
        decision["approved"] = True
        decision["destination"] = "Poems/ambiguous.txt"
        self.plan.write_text(json.dumps(plan), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 0)
        self.assertTrue((self.root / "Poems" / "ambiguous.txt").exists())
        self.assertTrue((self.root / "bill.txt").exists())

    def test_thresholds_are_not_invented_when_not_supplied(self):
        self.populate()
        plan, _ = self.preview(
            match_threshold=None, match_margin=None, new_group_threshold=None
        )
        self.assertTrue(all(item["classification"] == "review" for item in plan["decisions"]))
        self.assertTrue(all(item["destination"] is None for item in plan["decisions"]))
        invoice = next(item for item in plan["decisions"] if item["source"] == "bill.txt")
        self.assertIn("nearest folder is Invoices", invoice["reason"])
        self.assertAlmostEqual(invoice["score"], 1.0)

    def test_existing_cluster_folders_are_valid_categories(self):
        self.write("Cluster_9_receipts/reference.txt", "invoice accounting reference document")
        self.write("bill.txt", "invoice payment for the current month")
        plan, _ = self.preview(new_group_threshold=None)
        self.assertEqual(plan["decisions"][0]["classification"], "existing")
        self.assertEqual(
            plan["decisions"][0]["destination"], "Cluster_9_receipts/bill.txt"
        )

    def test_media_and_text_share_profiles_and_decode_failures_are_isolated(self):
        class MediaEmbedder:
            name = "media-topic-test"
            cacheable = False
            media_kinds = frozenset({"image"})

            def embed_corpus(self, texts):
                return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

            def embed_media(self, items):
                if any(path.name == "broken.png" for path, _kind in items):
                    raise ValueError("cannot decode")
                return np.asarray([[1.0, 0.0] for _ in items], dtype=np.float32)

        folder = self.root / "Invoices"
        folder.mkdir()
        (folder / "reference.png").write_bytes(b"fake reference image")
        (self.root / "receipt.png").write_bytes(b"fake incoming image")
        (self.root / "broken.png").write_bytes(b"cannot decode")
        self.write("bill.txt", "invoice payment for the current month")
        settings = ss.Settings(
            root=self.root, organize=True, match_threshold=0.9, match_margin=0.1,
            save_plan=self.plan, media=("image",), use_cache=False,
        )
        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="WARNING"):
            self.assertEqual(ss.run(settings, embedder=MediaEmbedder()), 0)
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        decisions = {item["source"]: item for item in plan["decisions"]}
        self.assertEqual(decisions["receipt.png"]["destination"], "Invoices/receipt.png")
        self.assertEqual(decisions["bill.txt"]["destination"], "Invoices/bill.txt")
        self.assertEqual(decisions["broken.png"]["classification"], "review")
        self.assertIn("could not be decoded", decisions["broken.png"]["reason"])

    def test_recursive_source_is_separate_from_reference_folders(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        self.write("Inbox/nested/bill.txt", "invoice payment for the current month")
        self.write("Inbox/Cluster_5/old.txt", "invoice already sorted document")
        self.write("loose.txt", "invoice loose file is not selected")
        plan, _ = self.preview(source=Path("Inbox"), recursive=True, new_group_threshold=None)
        self.assertEqual(plan["source"], "Inbox")
        self.assertEqual([item["source"] for item in plan["decisions"]], ["Inbox/nested/bill.txt"])
        self.assertEqual(
            plan["decisions"][0]["destination"], "Invoices/bill.txt"
        )
        self.assertNotIn("Inbox", [category["folder"] for category in plan["inputs"]["categories"]])

    def test_explicit_source_cannot_bypass_excluded_directories(self):
        for name in (".smart_sort", "node_modules", "Inbox/.hidden", "Cluster_1"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "excluded or hidden"):
                    ss._organization_inputs(
                        self.root, Path(name), True, ss.DEFAULT_EXCLUDES, None
                    )

    def test_recursive_duplicate_names_are_safely_disambiguated_and_undone(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        first = self.write("Inbox/a/report.txt", "invoice payment from source alpha")
        second = self.write("Inbox/b/report.txt", "invoice payment from source bravo")
        plan, _ = self.preview(source=Path("Inbox"), recursive=True, new_group_threshold=None)
        destinations = {item["destination"] for item in plan["decisions"]}
        self.assertEqual(destinations, {"Invoices/report.txt", "Invoices/report (2).txt"})
        self.approve(plan, {"Inbox/a/report.txt", "Inbox/b/report.txt"})
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 0)
        manifest = next(
            line.removeprefix("Manifest: ")
            for line in output.getvalue().splitlines()
            if line.startswith("Manifest: ")
        )
        self.assertEqual(ss.undo_manifest(Path(manifest), dry_run=False), (2, 0))
        self.assertEqual(first.read_text(encoding="utf-8"), "invoice payment from source alpha")
        self.assertEqual(second.read_text(encoding="utf-8"), "invoice payment from source bravo")

    def test_reference_sampling_is_bounded_and_reports_total(self):
        for index in range(30):
            self.write(f"Invoices/reference-{index:02}.txt", "invoice accounting reference")
        self.write("bill.txt", "invoice payment for the current month")
        plan, _ = self.preview(new_group_threshold=None)
        category = plan["inputs"]["categories"][0]
        self.assertEqual(category["folder"], "Invoices")
        self.assertEqual(category["total"], 30)
        self.assertEqual(len(category["files"]), 25)
        self.assertEqual(plan["decisions"][0]["classification"], "existing")

    def test_group_size_and_pairwise_threshold_leave_outliers_for_review(self):
        self.write("recipe-a.txt", "recipe ingredients and baking instructions")
        self.write("recipe-b.txt", "recipe cooking and dinner instructions")
        self.write("solo.txt", "travel destinations and vacation details")
        plan, _ = self.preview(
            match_threshold=None, match_margin=None, min_group_size=3
        )
        self.assertTrue(all(item["classification"] == "review" for item in plan["decisions"]))
        self.assertTrue(all(item["destination"] is None for item in plan["decisions"]))

    def test_plan_inside_source_is_excluded_from_input_snapshot(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        self.write("bill.txt", "invoice payment for the current month")
        self.plan = self.root / "review.json"
        plan, _ = self.preview(new_group_threshold=None)
        self.approve(plan, {"bill.txt"})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 0)
        self.assertTrue((self.root / "Invoices" / "bill.txt").exists())
        self.assertTrue(self.plan.exists())

    def test_existing_plan_is_never_overwritten(self):
        self.write("bill.txt", "invoice payment for the current month")
        self.plan.write_text("existing user content", encoding="utf-8")
        settings = ss.Settings(root=self.root, organize=True, save_plan=self.plan, use_cache=False)
        with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
            self.assertEqual(ss.run(settings, embedder=TopicEmbedder()), 1)
        self.assertEqual(self.plan.read_text(encoding="utf-8"), "existing user content")
        self.assertTrue((self.root / "bill.txt").exists())

    def test_source_change_after_locked_preflight_is_rejected(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        source = self.write("bill.txt", "invoice payment for the current month")
        plan, _ = self.preview(new_group_threshold=None)
        self.approve(plan, {"bill.txt"})
        actual = ss.apply_moves

        def mutate_after_preflight(root, moves, metadata, *, preflight):
            def mutate():
                preflight()
                source.write_text("changed after review and preflight", encoding="utf-8")
            return actual(root, moves, metadata, preflight=mutate)

        with mock.patch.object(ss, "apply_moves", side_effect=mutate_after_preflight):
            with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
                self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertEqual(source.read_text(encoding="utf-8"), "changed after review and preflight")
        self.assertFalse((self.root / "Invoices" / "bill.txt").exists())

    def test_plan_changed_before_locked_preflight_is_rejected(self):
        self.write("Invoices/reference.txt", "invoice accounting reference document")
        source = self.write("bill.txt", "invoice payment for the current month")
        plan, _ = self.preview(new_group_threshold=None)
        self.approve(plan, {"bill.txt"})
        actual = ss.apply_moves

        def mutate_plan(root, moves, metadata, *, preflight):
            self.plan.write_text(self.plan.read_text(encoding="utf-8") + " ", encoding="utf-8")
            return actual(root, moves, metadata, preflight=preflight)

        with mock.patch.object(ss, "apply_moves", side_effect=mutate_plan):
            with redirect_stdout(io.StringIO()), self.assertLogs("smart-sort", level="ERROR"):
                self.assertEqual(ss.apply_organize_plan(self.root, self.plan, apply=True), 1)
        self.assertTrue(source.exists())
        self.assertFalse((self.root / "Invoices" / "bill.txt").exists())

    def test_tfidf_cli_plan_approval_apply_and_undo(self):
        text = "invoice payment ledger finance accounting quarterly report"
        self.write("Invoices/reference.txt", text)
        self.write("bill.txt", text)
        output = io.StringIO()
        with redirect_stdout(output):
            result = ss.main(
                [
                    str(self.root), "--organize", "--backend", "tfidf",
                    "--match-threshold", "0.9", "--match-margin", "0.1",
                    "--save-plan", str(self.plan),
                ]
            )
        self.assertEqual(result, 0)
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        self.assertEqual(plan["decisions"][0]["destination"], "Invoices/bill.txt")
        self.approve(plan, {"bill.txt"})
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ss.main([str(self.root), "--apply-plan", str(self.plan)]), 0)
        self.assertTrue((self.root / "bill.txt").exists())

        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                ss.main([str(self.root), "--apply-plan", str(self.plan), "--apply"]), 0
            )
        manifest = next(
            line.removeprefix("Manifest: ")
            for line in output.getvalue().splitlines()
            if line.startswith("Manifest: ")
        )
        self.assertTrue((self.root / "Invoices" / "bill.txt").exists())
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ss.main(["--undo", manifest, "--apply"]), 0)
        self.assertTrue((self.root / "bill.txt").exists())

    def test_cli_rejects_unreviewed_or_invalid_modes(self):
        invalid = [
            [str(self.root), "--organize", "--apply"],
            [str(self.root), "--organize", "--recursive"],
            [str(self.root), "--organize", "--match-threshold", "0.5"],
            [str(self.root), "--organize", "--new-group-threshold", "nan"],
            [str(self.root), "--organize", "--min-group-size", "1"],
            [str(self.root), "--source", "Inbox"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    ss.main(arguments)
                self.assertEqual(exc.exception.code, 2)

    def test_no_labels_and_timings_apply_to_organizing(self):
        self.write("recipe-a.txt", "recipe ingredients and baking instructions")
        self.write("recipe-b.txt", "recipe cooking and dinner instructions")
        plan, output = self.preview(
            match_threshold=None, match_margin=None, label_clusters=False, timings=True
        )
        self.assertTrue(
            all(item["destination"].startswith("Cluster_") for item in plan["decisions"])
        )
        self.assertTrue(
            all("_recipe" not in item["destination"] for item in plan["decisions"])
        )
        self.assertIn("Timings:", output)
        for phase in ("collect", "fingerprint", "extract", "embed", "verify", "cluster"):
            self.assertIn(phase, output)


if __name__ == "__main__":
    unittest.main()
