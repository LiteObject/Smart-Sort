#!/usr/bin/env python3
"""Evaluate organizing suggestions against locally supplied held-out labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Sequence


def _pairs(count: int) -> int:
    return count * (count - 1) // 2


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metrics(cases: list[dict], decisions: dict[str, dict]) -> dict:
    predicted_existing = 0
    expected_existing = 0
    correct_existing = 0
    predicted_groups: Counter[str] = Counter()
    expected_groups: Counter[str] = Counter()
    intersections: Counter[tuple[str, str]] = Counter()

    for case in cases:
        decision = decisions[case["source"]]
        expected = case["expected"]
        prediction = decision["classification"]
        if expected == "existing":
            expected_existing += 1
        if prediction == "existing":
            predicted_existing += 1
            folder = PurePosixPath(decision["destination"]).parts[0]
            if expected == "existing" and folder == case["folder"]:
                correct_existing += 1

        if expected == "new":
            expected_groups[case["group"]] += 1
        if prediction == "new":
            folder = str(PurePosixPath(decision["destination"]).parent)
            predicted_groups[folder] += 1
            if expected == "new":
                intersections[(folder, case["group"])] += 1

    expected_pairs = sum(_pairs(count) for count in expected_groups.values())
    predicted_pairs = sum(_pairs(count) for count in predicted_groups.values())
    correct_pairs = sum(_pairs(count) for count in intersections.values())
    existing_precision = _ratio(correct_existing, predicted_existing)
    existing_recall = _ratio(correct_existing, expected_existing)
    new_precision = _ratio(correct_pairs, predicted_pairs)
    new_recall = _ratio(correct_pairs, expected_pairs)
    targets_met = (
        existing_precision is not None
        and existing_precision >= 0.95
        and existing_recall is not None
        and existing_recall >= 0.70
        and new_precision is not None
        and new_precision >= 0.90
        and new_recall is not None
        and new_recall >= 0.50
    )
    return {
        "files": len(cases),
        "existing": {
            "correct": correct_existing,
            "predicted": predicted_existing,
            "expected": expected_existing,
            "precision": existing_precision,
            "recall": existing_recall,
        },
        "new_groups": {
            "correct_pairs": correct_pairs,
            "predicted_pairs": predicted_pairs,
            "expected_pairs": expected_pairs,
            "pairwise_precision": new_precision,
            "pairwise_recall": new_recall,
        },
        "targets_met": targets_met,
    }


def evaluate_plan(plan: dict, labels: dict) -> dict:
    if not isinstance(plan, dict) or plan.get("type") != "smart-sort-plan" or plan.get("version") != 1:
        raise ValueError("Not a supported Smart-Sort organizing plan")
    if not isinstance(labels, dict) or labels.get("type") != "smart-sort-labels" or labels.get("version") != 1:
        raise ValueError("Not a supported Smart-Sort labels file")
    if labels.get("split") != "held-out":
        raise ValueError("Labels must identify the held-out evaluation split")
    dataset_id = labels.get("dataset_id")
    calibration_ids = labels.get("calibration_case_ids")
    cases = labels.get("cases")
    raw_decisions = plan.get("decisions")
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(calibration_ids, list)
        or not calibration_ids
        or not all(isinstance(item, str) and item for item in calibration_ids)
        or len(set(calibration_ids)) != len(calibration_ids)
        or not isinstance(cases, list)
        or not isinstance(raw_decisions, list)
        or not isinstance(plan.get("backend"), str)
    ):
        raise ValueError("Labels or plan are missing required evaluation metadata")

    decisions: dict[str, dict] = {}
    for decision in raw_decisions:
        if not isinstance(decision, dict) or decision.get("approved") is not False:
            raise ValueError("Benchmark an unapproved preview, not a manually approved plan")
        source = decision.get("source")
        classification = decision.get("classification")
        if (
            not isinstance(source, str)
            or not source
            or source in decisions
            or classification not in {"existing", "new", "review"}
        ):
            raise ValueError("Preview has an invalid or duplicate decision")
        destination = decision.get("destination")
        if classification == "review":
            if destination is not None:
                raise ValueError("Review decisions must not have manually assigned destinations")
        else:
            if not isinstance(destination, str):
                raise ValueError("A predicted folder requires a destination")
            relative = PurePosixPath(destination)
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
                raise ValueError("Prediction contains an invalid destination")
        decisions[source] = decision

    case_ids: set[str] = set()
    sources: set[str] = set()
    folders: set[str] = set()
    negatives = 0
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each label must be an object")
        case_id, source = case.get("case_id"), case.get("source")
        expected, kind = case.get("expected"), case.get("kind")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
            or not isinstance(source, str)
            or not source
            or source in sources
            or expected not in {"existing", "new", "unrelated", "ambiguous"}
            or kind not in {"text", "image", "audio", "video"}
        ):
            raise ValueError("Labels contain an invalid or duplicate case")
        case_ids.add(case_id)
        sources.add(source)
        if expected == "existing":
            folder = case.get("folder")
            if (
                not isinstance(folder, str)
                or not folder
                or folder in {".", ".."}
                or len(PurePosixPath(folder).parts) != 1
            ):
                raise ValueError("An existing-folder label must name a top-level folder")
            folders.add(folder)
        elif expected == "new":
            group = case.get("group")
            if not isinstance(group, str) or not group:
                raise ValueError("A new-group label must name its expected group")
        else:
            negatives += 1
    if case_ids.intersection(calibration_ids):
        raise ValueError("Calibration and held-out case IDs overlap")
    if sources != set(decisions):
        raise ValueError("Labels and preview must contain exactly the same sources")

    overall = _metrics(cases, decisions)
    sufficient = len(cases) >= 200 and len(folders) >= 10 and negatives >= 40
    modalities = {
        kind: _metrics([case for case in cases if case["kind"] == kind], decisions)
        for kind in sorted({case["kind"] for case in cases})
    }
    return {
        "dataset_id": dataset_id,
        "backend": plan["backend"],
        "thresholds": plan.get("thresholds"),
        "calibration_cases": len(calibration_ids),
        "held_out_files": len(cases),
        "existing_categories": len(folders),
        "unrelated_or_ambiguous": negatives,
        "sample_sufficient": sufficient,
        "overall": overall,
        "modalities": modalities,
        "passed": sufficient and overall["targets_met"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate an unapproved organizing preview against held-out labels."
    )
    parser.add_argument("--plan", type=Path, required=True, help="Unapproved organizing plan.")
    parser.add_argument("--labels", type=Path, required=True, help="Held-out labels JSON.")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args(argv)
    try:
        with args.plan.open("r", encoding="utf-8") as handle:
            plan = json.load(handle)
        with args.labels.open("r", encoding="utf-8") as handle:
            labels = json.load(handle)
        report = evaluate_plan(plan, labels)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Dataset: {report['dataset_id']}  Backend: {report['backend']}")
        print(
            f"Held out: {report['held_out_files']} files, "
            f"{report['existing_categories']} existing categories, "
            f"{report['unrelated_or_ambiguous']} unrelated/ambiguous"
        )
        for kind, metrics in [("overall", report["overall"]), *report["modalities"].items()]:
            existing, new = metrics["existing"], metrics["new_groups"]
            print(
                f"{kind}: existing precision={existing['precision']} recall={existing['recall']}; "
                f"new pairwise precision={new['pairwise_precision']} recall={new['pairwise_recall']}"
            )
        print("Targets passed." if report["passed"] else "Targets not met or insufficient data.")
        print("The caller must ensure labels are representative and case IDs identify distinct files.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
