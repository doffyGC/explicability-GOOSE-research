"""Tests for the checklist E.4/E.5 prediction-integrity audit."""

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from check_prediction_integrity import (
    build_report,
    check_class_sums,
    check_row_coverage,
    check_scores_consistency,
    confusion_counts,
    load_prediction_arrays,
    main,
    metrics_from_confusion,
    pair_runs,
)
from run_grouped_validation import ScoreWriter

CLASSES = ["attack", "normal"]


def write_run(directory, rows, fold_metrics, status="full_grouped_run", rows_used=None):
    """Materialise a minimal run directory: report JSON + predictions CSV."""
    os.makedirs(directory, exist_ok=True)
    report = {
        "status": status,
        "dataset": os.path.join(directory, "does-not-exist.parquet"),
        "dataset_sha256": "0" * 64,
        "target_column": "class",
        "classes": CLASSES,
        "rows_used": len(rows) if rows_used is None else rows_used,
        "fold_metrics": fold_metrics,
    }
    with open(os.path.join(directory, "grouped_validation_report.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(directory, "grouped_predictions.csv"), "w",
              encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["split_id", "row_index", "split_group", "y_true", "y_pred"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return directory


def consistent_rows():
    """8 rows over 2 folds, with the fold metrics they actually imply."""
    rows = [
        # fold-00: true normal,normal,attack,attack / pred normal,normal,normal,attack
        {"split_id": "fold-00", "row_index": 0, "split_group": "R1", "y_true": "normal", "y_pred": "normal"},
        {"split_id": "fold-00", "row_index": 1, "split_group": "R1", "y_true": "normal", "y_pred": "normal"},
        {"split_id": "fold-00", "row_index": 2, "split_group": "R1", "y_true": "attack", "y_pred": "normal"},
        {"split_id": "fold-00", "row_index": 3, "split_group": "R1", "y_true": "attack", "y_pred": "attack"},
        # fold-01: true normal,normal,normal,attack / pred all normal
        {"split_id": "fold-01", "row_index": 4, "split_group": "R2", "y_true": "normal", "y_pred": "normal"},
        {"split_id": "fold-01", "row_index": 5, "split_group": "R2", "y_true": "normal", "y_pred": "normal"},
        {"split_id": "fold-01", "row_index": 6, "split_group": "R2", "y_true": "normal", "y_pred": "normal"},
        {"split_id": "fold-01", "row_index": 7, "split_group": "R2", "y_true": "attack", "y_pred": "normal"},
    ]
    f1_attack_0 = 2 * 1.0 * 0.5 / 1.5
    f1_normal_0 = 2 * (2 / 3) * 1.0 / ((2 / 3) + 1.0)
    f1_normal_1 = 2 * 0.75 * 1.0 / 1.75
    fold_metrics = [
        {
            "split_id": "fold-00", "train_rows": 4, "test_rows": 4,
            "accuracy": 0.75,
            "macro_f1": (f1_attack_0 + f1_normal_0) / 2,
            "weighted_f1": (f1_attack_0 * 2 + f1_normal_0 * 2) / 4,
            "per_class": {
                "attack": {"precision": 1.0, "recall": 0.5, "f1-score": f1_attack_0, "support": 2},
                "normal": {"precision": 2 / 3, "recall": 1.0, "f1-score": f1_normal_0, "support": 2},
            },
        },
        {
            "split_id": "fold-01", "train_rows": 4, "test_rows": 4,
            "accuracy": 0.75,
            "macro_f1": (0.0 + f1_normal_1) / 2,
            "weighted_f1": (0.0 * 1 + f1_normal_1 * 3) / 4,
            "per_class": {
                "attack": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 1},
                "normal": {"precision": 0.75, "recall": 1.0, "f1-score": f1_normal_1, "support": 3},
            },
        },
    ]
    return rows, fold_metrics


class MetricTests(unittest.TestCase):
    def test_per_class_macro_and_weighted_are_computed_separately(self):
        y_true = np.array([1, 1, 0, 0, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 1, 1])
        result = metrics_from_confusion(confusion_counts(y_true, y_pred, 2), CLASSES)

        self.assertAlmostEqual(result["accuracy"], 6 / 8)
        self.assertEqual(result["per_class"]["attack"]["support"], 3)
        self.assertAlmostEqual(result["per_class"]["attack"]["recall"], 1 / 3)
        self.assertAlmostEqual(result["per_class"]["normal"]["recall"], 1.0)
        # Macro and weighted must not coincide when supports differ.
        self.assertNotAlmostEqual(
            result["averages"]["macro"]["f1-score"],
            result["averages"]["weighted"]["f1-score"],
        )

    def test_matches_sklearn_on_a_random_multiclass_sample(self):
        """The audit is only independent if it is also correct."""
        from sklearn.metrics import classification_report

        rng = np.random.RandomState(7)
        names = ["a", "b", "c", "d"]
        y_true = rng.randint(0, 4, size=5000)
        y_pred = rng.randint(0, 4, size=5000)
        # Force one class to be never predicted: the zero-division path.
        y_pred[y_pred == 3] = 0

        mine = metrics_from_confusion(confusion_counts(y_true, y_pred, 4), names)
        theirs = classification_report(
            y_true, y_pred, labels=list(range(4)), target_names=names,
            output_dict=True, zero_division=0,
        )
        for scheme in ("macro", "weighted"):
            for key in ("precision", "recall", "f1-score"):
                self.assertAlmostEqual(
                    mine["averages"][scheme][key], theirs["%s avg" % scheme][key], places=12,
                    msg="%s avg %s" % (scheme, key))
        for name in names:
            for key in ("precision", "recall", "f1-score"):
                self.assertAlmostEqual(mine["per_class"][name][key], theirs[name][key], places=12)
            self.assertEqual(mine["per_class"][name]["support"], theirs[name]["support"])


class CoverageTests(unittest.TestCase):
    @staticmethod
    def arrays(row_index):
        sorted_index = np.sort(np.asarray(row_index, dtype=np.int64))
        return {"row_index": sorted_index}

    def test_duplicate_row_is_caught(self):
        checks = check_row_coverage(
            self.arrays([0, 1, 1, 2]),
            {"status": "full_grouped_run", "rows_used": 4},
        )
        duplicate_check = next(c for c in checks if "at most once" in c["check"])
        self.assertFalse(duplicate_check["passed"])

    def test_gap_in_a_full_run_is_caught(self):
        checks = check_row_coverage(
            self.arrays([0, 1, 2, 5]),
            {"status": "full_grouped_run", "rows_used": 4},
        )
        coverage_check = next(c for c in checks if "no gaps" in c["check"])
        self.assertFalse(coverage_check["passed"])

    def test_complete_full_run_passes(self):
        checks = check_row_coverage(
            self.arrays([0, 1, 2, 3]),
            {"status": "full_grouped_run", "rows_used": 4},
        )
        self.assertTrue(all(c["passed"] for c in checks))


class ClassSumTests(unittest.TestCase):
    def test_support_disagreement_is_caught(self):
        arrays = {"y_true": np.array([0, 0, 1, 1, 1])}
        report = {"fold_metrics": [{"per_class": {
            "attack": {"support": 2}, "normal": {"support": 4},  # one too many
        }}]}
        checks = check_class_sums(arrays, report, CLASSES)
        normal_check = next(c for c in checks if c["check"].endswith("normal"))
        self.assertFalse(normal_check["passed"])

    def test_capped_smoke_run_may_predict_fewer_rows_than_the_dataset(self):
        arrays = {"y_true": np.array([0, 1, 1])}
        report = {
            "status": "technical_smoke",
            "fold_metrics": [{"per_class": {
                "attack": {"support": 1}, "normal": {"support": 2},
            }}],
        }
        checks = check_class_sums(arrays, report, CLASSES,
                                  dataset_counts={"attack": 10, "normal": 90})
        self.assertTrue(all(c["passed"] for c in checks))

    def test_full_run_must_match_the_dataset_exactly(self):
        arrays = {"y_true": np.array([0, 1, 1])}
        report = {
            "status": "full_grouped_run",
            "fold_metrics": [{"per_class": {
                "attack": {"support": 1}, "normal": {"support": 2},
            }}],
        }
        checks = check_class_sums(arrays, report, CLASSES,
                                  dataset_counts={"attack": 10, "normal": 90})
        self.assertFalse(all(c["passed"] for c in checks))


class PairingTests(unittest.TestCase):
    @staticmethod
    def run_with(row_index, y_true, y_pred):
        return {"arrays": {
            "row_index": np.asarray(row_index, dtype=np.int64),
            "y_true": np.asarray(y_true, dtype=np.int16),
            "y_pred": np.asarray(y_pred, dtype=np.int16),
        }}

    def test_same_rows_give_a_mcnemar_ready_table(self):
        first = self.run_with([0, 1, 2, 3], [1, 1, 0, 0], [1, 0, 0, 1])
        second = self.run_with([0, 1, 2, 3], [1, 1, 0, 0], [1, 1, 1, 1])
        result = pair_runs(first, second)
        self.assertTrue(result["paired"])
        self.assertEqual(result["n_paired"], 4)
        self.assertEqual(result["both_correct"], 1)     # row 0
        self.assertEqual(result["only_first_correct"], 1)   # row 2
        self.assertEqual(result["only_second_correct"], 1)  # row 1
        self.assertEqual(result["neither_correct"], 1)  # row 3
        self.assertEqual(result["discordant"], 2)

    def test_different_rows_are_not_pairable(self):
        first = self.run_with([0, 1, 2], [1, 1, 0], [1, 1, 0])
        second = self.run_with([0, 1, 9], [1, 1, 0], [1, 1, 0])
        result = pair_runs(first, second)
        self.assertFalse(result["paired"])
        self.assertEqual(result["n_paired"], 0)

    def test_same_rows_but_different_ground_truth_is_not_pairable(self):
        first = self.run_with([0, 1], [1, 1], [1, 1])
        second = self.run_with([0, 1], [1, 0], [1, 1])
        result = pair_runs(first, second)
        self.assertFalse(result["paired"])
        self.assertFalse(result["same_y_true"])
        self.assertTrue(result["same_rows"])


class CliTests(unittest.TestCase):
    def test_consistent_run_passes_and_writes_a_report(self):
        rows, fold_metrics = consistent_rows()
        with tempfile.TemporaryDirectory() as temp:
            run_dir = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            out = os.path.join(temp, "audit.md")
            code = main(["--run", run_dir, "--out", out, "--skip-dataset-check"])
            self.assertEqual(code, 0)
            with open(out, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("macro F1", text)
            self.assertIn("weighted F1", text)

    def test_json_out_is_written_and_serializable(self):
        """Regression: recomputed metrics are numpy scalars until cast."""
        rows, fold_metrics = consistent_rows()
        with tempfile.TemporaryDirectory() as temp:
            run_dir = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            json_out = os.path.join(temp, "audit.json")
            code = main(["--run", run_dir, "--out", os.path.join(temp, "audit.md"),
                         "--json-out", json_out, "--skip-dataset-check"])
            self.assertEqual(code, 0)
            with open(json_out, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            pooled = payload["runs"][0]["pooled"]
            self.assertIsInstance(pooled["accuracy"], float)
            self.assertIn("weighted", pooled["averages"])
            self.assertTrue(all(isinstance(c["passed"], bool)
                                for c in payload["runs"][0]["checks"]))

    def test_tampered_support_fails_with_exit_one(self):
        rows, fold_metrics = consistent_rows()
        fold_metrics[0]["per_class"]["attack"]["support"] = 5  # never true
        with tempfile.TemporaryDirectory() as temp:
            run_dir = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            code = main(["--run", run_dir, "--out", os.path.join(temp, "audit.md"),
                         "--skip-dataset-check"])
            self.assertEqual(code, 1)

    def test_tampered_metric_fails_with_exit_one(self):
        rows, fold_metrics = consistent_rows()
        fold_metrics[1]["macro_f1"] = 0.99  # not what these predictions produce
        with tempfile.TemporaryDirectory() as temp:
            run_dir = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            code = main(["--run", run_dir, "--out", os.path.join(temp, "audit.md"),
                         "--skip-dataset-check"])
            self.assertEqual(code, 1)

    def test_duplicate_prediction_row_fails(self):
        rows, fold_metrics = consistent_rows()
        rows[1]["row_index"] = 0  # same row predicted twice
        with tempfile.TemporaryDirectory() as temp:
            run_dir = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            code = main(["--run", run_dir, "--out", os.path.join(temp, "audit.md"),
                         "--skip-dataset-check"])
            self.assertEqual(code, 1)

    def test_two_runs_over_the_same_rows_are_reported_as_pairable(self):
        rows, fold_metrics = consistent_rows()
        with tempfile.TemporaryDirectory() as temp:
            first = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            second = write_run(os.path.join(temp, "run-b"), rows, fold_metrics)
            out = os.path.join(temp, "audit.md")
            code = main(["--run", first, "--run", second, "--out", out,
                         "--skip-dataset-check"])
            self.assertEqual(code, 0)
            with open(out, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("| `run-a` | `run-b` | yes |", text)

    def test_runs_over_different_rows_are_refused(self):
        rows, fold_metrics = consistent_rows()
        shifted = [dict(row, row_index=row["row_index"] + 100) for row in rows]
        with tempfile.TemporaryDirectory() as temp:
            first = write_run(os.path.join(temp, "run-a"), rows, fold_metrics)
            second = write_run(os.path.join(temp, "run-b"), shifted, fold_metrics,
                               status="technical_smoke")
            code = main(["--run", first, "--run", second,
                         "--out", os.path.join(temp, "audit.md"), "--skip-dataset-check"])
            self.assertEqual(code, 1)


class ReportTests(unittest.TestCase):
    def test_single_run_report_states_no_pairing_instead_of_omitting_it(self):
        pooled = metrics_from_confusion(
            confusion_counts(np.array([0, 1]), np.array([0, 1]), 2), CLASSES)
        runs = [{
            "label": "run-a",
            "report": {"status": "full_grouped_run", "model": "decision-tree", "balance": "none"},
            "fold_ids": ["fold-00"],
            "checks": [{"check": "x", "expected": "1", "observed": "1", "passed": True}],
            "pooled": pooled,
        }]
        text = "\n".join(build_report(runs, []))
        self.assertIn("single run audited", text)


if __name__ == "__main__":
    unittest.main()


def write_scores(directory, rows, posteriors, classes=CLASSES):
    """Persist posteriors for a run built by `write_run`, via the real writer."""
    groups = sorted({row["split_group"] for row in rows})
    group_index = {label: index for index, label in enumerate(groups)}
    class_index = {name: index for index, name in enumerate(classes)}
    path = os.path.join(directory, "grouped_scores.parquet")
    writer = ScoreWriter(path, classes, groups)
    try:
        folds = []
        for row in rows:
            if row["split_id"] not in folds:
                folds.append(row["split_id"])
        for fold in folds:
            picked = [index for index, row in enumerate(rows)
                      if rows[index]["split_id"] == fold]
            writer.write_fold(
                fold,
                np.asarray([rows[i]["row_index"] for i in picked], dtype="int64"),
                np.asarray([group_index[rows[i]["split_group"]] for i in picked],
                           dtype="int32"),
                np.asarray([class_index[rows[i]["y_true"]] for i in picked],
                           dtype="int32"),
                np.asarray([posteriors[i] for i in picked], dtype="float32"),
            )
    finally:
        writer.close()
    return path


def posteriors_agreeing_with(rows, classes=CLASSES):
    """A posterior per row whose argmax is that row's recorded `y_pred`."""
    out = []
    for row in rows:
        vector = [0.1] * len(classes)
        vector[classes.index(row["y_pred"])] = 0.9
        total = sum(vector)
        out.append([value / total for value in vector])
    return out


class ScoresConsistencyTests(unittest.TestCase):
    """The audit `run_grouped_validation.py` is not allowed to do for itself.

    The runner checks its own argmax against `model.predict` on the first
    block of each fold only. These tests cover the independent, full-coverage
    re-derivation from the file on disk.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rows, self.fold_metrics = consistent_rows()
        self.directory = write_run(os.path.join(self.tmp.name, "run"),
                                   self.rows, self.fold_metrics)
        self.report = json.load(open(
            os.path.join(self.directory, "grouped_validation_report.json"),
            encoding="utf-8"))
        self.arrays = load_prediction_arrays(
            os.path.join(self.directory, "grouped_predictions.csv"),
            CLASSES, [f["split_id"] for f in self.fold_metrics])

    def _checks(self):
        return check_scores_consistency(self.directory, self.arrays,
                                        self.report, CLASSES)

    def test_absent_scores_file_produces_no_checks(self):
        # --save-scores is optional; an unscored run is not a broken run.
        self.assertEqual(self._checks(), [])

    def test_agreeing_posteriors_pass_every_check(self):
        write_scores(self.directory, self.rows, posteriors_agreeing_with(self.rows))
        checks = self._checks()
        self.assertEqual(len(checks), 4)
        self.assertTrue(all(check["passed"] for check in checks), checks)

    def test_a_single_disagreeing_row_is_caught(self):
        posteriors = posteriors_agreeing_with(self.rows)
        # Row 3 was predicted `attack`; flip its posterior to favour `normal`.
        posteriors[3] = [0.1, 0.9]
        write_scores(self.directory, self.rows, posteriors)
        failed = [c for c in self._checks() if not c["passed"]]
        self.assertEqual(len(failed), 1)
        self.assertIn("argmax(posterior) reproduces y_pred", failed[0]["check"])
        self.assertIn("1 mismatches", failed[0]["observed"])

    def test_posteriors_that_are_not_a_distribution_are_caught(self):
        posteriors = posteriors_agreeing_with(self.rows)
        # Sums to 1.1, but its argmax still matches row 0's `normal`, so only
        # the distribution check may fire - a broken block must not be able to
        # hide behind a correct argmax.
        posteriors[0] = [0.2, 0.9]
        write_scores(self.directory, self.rows, posteriors)
        failed = [c for c in self._checks() if not c["passed"]]
        self.assertEqual([c["check"] for c in failed],
                         ["posteriors are a distribution (in [0,1], summing to 1)"])

    def test_missing_rows_are_caught(self):
        write_scores(self.directory, self.rows[:6],
                     posteriors_agreeing_with(self.rows)[:6])
        failed = [c for c in self._checks() if not c["passed"]]
        self.assertEqual([c["check"] for c in failed], ["scored rows == predicted rows"])
        self.assertIn("6 scored", failed[0]["observed"])

    def test_disagreeing_ground_truth_is_caught(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["y_true"] = "attack"
        write_scores(self.directory, rows, posteriors_agreeing_with(self.rows))
        failed = [c for c in self._checks() if not c["passed"]]
        self.assertEqual([c["check"] for c in failed],
                         ["scores y_true agrees with predictions y_true"])
