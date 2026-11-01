"""Tests for held-out sorting metrics and sample-size gates."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import benchmark_sort as benchmark
import smart_sort as ss


def synthetic_evaluation():
    cases = []
    decisions = []
    for folder_number in range(10):
        for index in range(12):
            source = f"existing-{folder_number}-{index}.txt"
            cases.append(
                {
                    "case_id": f"held-{source}",
                    "source": source,
                    "kind": "text" if index % 2 == 0 else "image",
                    "expected": "existing",
                    "folder": f"Category_{folder_number}",
                }
            )
            decisions.append(
                {
                    "source": source,
                    "classification": "existing",
                    "destination": f"Category_{folder_number}/{source}",
                    "approved": False,
                }
            )
    for group_number in range(5):
        for index in range(8):
            source = f"new-{group_number}-{index}.txt"
            cases.append(
                {
                    "case_id": f"held-{source}",
                    "source": source,
                    "kind": "text" if index % 2 == 0 else "image",
                    "expected": "new",
                    "group": f"Novel_{group_number}",
                }
            )
            decisions.append(
                {
                    "source": source,
                    "classification": "new",
                    "destination": f"Cluster_{group_number}/{source}",
                    "approved": False,
                }
            )
    for index in range(40):
        source = f"negative-{index}.txt"
        cases.append(
            {
                "case_id": f"held-{source}",
                "source": source,
                "kind": "text" if index % 2 == 0 else "image",
                "expected": "unrelated" if index < 20 else "ambiguous",
            }
        )
        decisions.append(
            {
                "source": source,
                "classification": "review",
                "destination": None,
                "approved": False,
            }
        )
    plan = {
        "type": "smart-sort-plan",
        "version": 1,
        "backend": "synthetic",
        "thresholds": {"match": 0.6, "margin": 0.1, "new_group": 0.8},
        "decisions": decisions,
    }
    labels = {
        "type": "smart-sort-labels",
        "version": 1,
        "dataset_id": "synthetic-held-out",
        "split": "held-out",
        "calibration_case_ids": ["calibration-1", "calibration-2"],
        "cases": cases,
    }
    return plan, labels


class BenchmarkTestCase(unittest.TestCase):
    def test_exact_metrics_and_required_sample_shape(self):
        plan, labels = synthetic_evaluation()
        report = benchmark.evaluate_plan(plan, labels)
        self.assertEqual(report["held_out_files"], 200)
        self.assertEqual(report["existing_categories"], 10)
        self.assertEqual(report["unrelated_or_ambiguous"], 40)
        self.assertTrue(report["sample_sufficient"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["overall"]["existing"]["precision"], 1.0)
        self.assertEqual(report["overall"]["existing"]["recall"], 1.0)
        self.assertEqual(report["overall"]["new_groups"]["pairwise_precision"], 1.0)
        self.assertEqual(report["overall"]["new_groups"]["pairwise_recall"], 1.0)
        self.assertEqual(report["overall"]["new_groups"]["expected_pairs"], 140)
        self.assertEqual(set(report["modalities"]), {"text", "image"})
        self.assertEqual(report["modalities"]["text"]["files"], 100)

    def test_false_existing_predictions_lower_precision_below_gate(self):
        plan, labels = synthetic_evaluation()
        for decision in plan["decisions"][-10:]:
            decision["classification"] = "existing"
            decision["destination"] = f"Category_0/{decision['source']}"
        report = benchmark.evaluate_plan(plan, labels)
        self.assertEqual(report["overall"]["existing"]["correct"], 120)
        self.assertEqual(report["overall"]["existing"]["predicted"], 130)
        self.assertAlmostEqual(report["overall"]["existing"]["precision"], 120 / 130)
        self.assertFalse(report["overall"]["targets_met"])
        self.assertFalse(report["passed"])

    def test_existing_folder_labels_are_not_conflated_by_case(self):
        plan, labels = synthetic_evaluation()
        labels["cases"][0]["folder"] = "category_0"
        report = benchmark.evaluate_plan(plan, labels)
        self.assertEqual(report["overall"]["existing"]["correct"], 119)
        self.assertEqual(report["existing_categories"], 11)

    def test_new_group_recall_measures_missing_pairs(self):
        plan, labels = synthetic_evaluation()
        for decision in plan["decisions"][122:160]:
            decision["classification"] = "review"
            decision["destination"] = None
        report = benchmark.evaluate_plan(plan, labels)
        metrics = report["overall"]["new_groups"]
        self.assertEqual(metrics["expected_pairs"], 140)
        self.assertEqual(metrics["correct_pairs"], 1)
        self.assertAlmostEqual(metrics["pairwise_recall"], 1 / 140)
        self.assertFalse(report["passed"])

    def test_insufficient_sample_does_not_pass_even_with_perfect_scores(self):
        plan, labels = synthetic_evaluation()
        plan["decisions"].pop()
        labels["cases"].pop()
        report = benchmark.evaluate_plan(plan, labels)
        self.assertEqual(report["held_out_files"], 199)
        self.assertTrue(report["overall"]["targets_met"])
        self.assertFalse(report["sample_sufficient"])
        self.assertFalse(report["passed"])

    def test_calibration_overlap_and_approved_plans_are_rejected(self):
        plan, labels = synthetic_evaluation()
        labels["calibration_case_ids"].append(labels["cases"][0]["case_id"])
        with self.assertRaisesRegex(ValueError, "overlap"):
            benchmark.evaluate_plan(plan, labels)
        labels["calibration_case_ids"].pop()
        plan["decisions"][0]["approved"] = True
        with self.assertRaisesRegex(ValueError, "unapproved"):
            benchmark.evaluate_plan(plan, labels)

    def test_cli_prints_a_machine_readable_report(self):
        plan, labels = synthetic_evaluation()
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp) / "plan.json"
            labels_path = Path(tmp) / "labels.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            labels_path.write_text(json.dumps(labels), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = benchmark.main(["--plan", str(plan_path), "--labels", str(labels_path), "--json"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())["passed"])

    def test_synthetic_200_file_tfidf_pipeline_meets_gate(self):
        cases = []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            root.mkdir()
            plan_path = base / "plan.json"

            def add(source, content, expected, **extra):
                (root / source).write_text(content, encoding="utf-8")
                cases.append(
                    {
                        "case_id": f"held-{source}",
                        "source": source,
                        "kind": "text",
                        "expected": expected,
                        **extra,
                    }
                )

            for index in range(10):
                folder = root / f"Category_{index:02}"
                folder.mkdir()
                (folder / "reference.txt").write_text(f"topic{index:02}", encoding="utf-8")
                for number in range(12):
                    add(
                        f"existing-{index:02}-{number:02}.txt",
                        f"topic{index:02}",
                        "existing",
                        folder=folder.name,
                    )
            for index in range(5):
                for number in range(8):
                    add(
                        f"new-{index:02}-{number:02}.txt",
                        f"noveltopic{index:02}",
                        "new",
                        group=f"Novel_{index}",
                    )
            for index in range(20):
                add(f"unknown-{index:02}.txt", f"unrelatedtoken{index:02}", "unrelated")
            for offset in (1, 2):
                for index in range(10):
                    add(
                        f"ambiguous-{offset}-{index:02}.txt",
                        f"topic{index:02} topic{(index + offset) % 10:02}",
                        "ambiguous",
                    )
            settings = ss.Settings(
                root=root,
                backend="tfidf",
                organize=True,
                match_threshold=0.25,
                match_margin=0.01,
                new_group_threshold=0.95,
                min_chars=1,
                use_cache=False,
                save_plan=plan_path,
            )
            with redirect_stdout(io.StringIO()):
                self.assertEqual(ss.run(settings), 0)
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            labels = {
                "type": "smart-sort-labels",
                "version": 1,
                "dataset_id": "synthetic-integration",
                "split": "held-out",
                "calibration_case_ids": ["calibration-only"],
                "cases": cases,
            }
            report = benchmark.evaluate_plan(plan, labels)
        self.assertEqual(report["held_out_files"], 200)
        self.assertEqual(report["existing_categories"], 10)
        self.assertEqual(report["unrelated_or_ambiguous"], 40)
        self.assertTrue(report["passed"])
        self.assertEqual(report["overall"]["existing"]["correct"], 120)
        self.assertEqual(report["overall"]["new_groups"]["correct_pairs"], 140)


if __name__ == "__main__":
    unittest.main()
