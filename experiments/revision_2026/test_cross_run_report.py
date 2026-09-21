"""Tests for `cross_run_report.py` (checklist D.5).

The consolidation script computes nothing new - it reconstructs pooled counts
from a per-fold block and puts 24 runs in one table. That makes two things
worth testing and nothing else very interesting:

1. **Is the reconstruction the same arithmetic the audited path uses?** It is
   checked against `check_prediction_integrity.metrics_from_confusion` on
   synthetic matrices, and against the published `run_bootstrap.champion_v2.md`
   point estimates, which were computed the expensive way from 11M rows.
2. **Does the table refuse rows that cannot share it?** A cross-run table
   looks equally authoritative whether or not its rows share a pool, and this
   revision regenerated its pool once. Most of the tests below are refusals.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest

import numpy as np

from check_prediction_integrity import metrics_from_confusion
from cross_run_report import (
    COUNT_TOLERANCE,
    CrossRunError,
    check_comparable,
    classify_run,
    designated_attack_class,
    load_average_precisions,
    load_run,
    metrics_from_counts,
    pooled_accuracy,
    pooled_counts,
)
from cross_run_report import main as cross_run_main

HERE = os.path.dirname(os.path.abspath(__file__))

ATTACK = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"
OTHER = "FULLY_RANDOMIZED_ORIENTEDGRAYHOLE"
CLASSES = [ATTACK, OTHER, "normal"]
DIGEST = "a" * 64


def fold_from_matrix(matrix, classes, split_id):
    """A per-fold report block carrying exactly what a confusion matrix says.

    This is the inverse of what the script does, so a round trip through it
    is a real test of the reconstruction rather than a restatement of it.
    """
    truth = metrics_from_confusion(np.asarray(matrix, dtype=np.int64), classes)
    return {
        "split_id": split_id,
        "train_rows": 0,
        "test_rows": int(np.asarray(matrix).sum()),
        "accuracy": truth["accuracy"],
        "macro_f1": truth["averages"]["macro"]["f1-score"],
        "weighted_f1": truth["averages"]["weighted"]["f1-score"],
        "averages": truth["averages"],
        "per_class": truth["per_class"],
    }


def make_report(matrices, classes=None, **overrides):
    classes = classes or CLASSES
    report = {
        "classes": list(classes),
        "dataset_sha256": DIGEST,
        "protocol": "stratified-group-kfold",
        "splits": "/somewhere/splits_grouped.json",
        "status": "full_grouped_run",
        "model": "xgboost",
        "balance": "none",
        "fold_metrics": [fold_from_matrix(m, classes, "fold-%02d" % i)
                         for i, m in enumerate(matrices)],
    }
    report.update(overrides)
    return report


def write_run(directory, report, name="run"):
    path = os.path.join(directory, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "grouped_validation_report.json"), "w",
              encoding="utf-8") as handle:
        json.dump(report, handle)
    return path


class ReconstructionTests(unittest.TestCase):
    """The pooled counts recovered from per-fold recall/precision/support."""

    # Two folds whose union is the quantity the table reports. Deliberately
    # lopsided: one fold carries most of the attack rows, which is the case
    # a fold mean gets wrong and pooling gets right.
    FOLD_A = [[80, 10, 10],
              [5, 40, 5],
              [3, 2, 1000]]
    FOLD_B = [[5, 1, 4],
              [0, 8, 2],
              [1, 1, 500]]

    def test_pooled_metrics_match_the_audited_implementation(self):
        report = make_report([self.FOLD_A, self.FOLD_B])
        classes, totals, residual = pooled_counts(report)
        pooled = metrics_from_counts(classes, totals)

        combined = np.asarray(self.FOLD_A) + np.asarray(self.FOLD_B)
        truth = metrics_from_confusion(combined, CLASSES)

        self.assertLess(residual, COUNT_TOLERANCE)
        for name in CLASSES:
            for key in ("precision", "recall", "f1-score", "support"):
                self.assertAlmostEqual(
                    pooled["per_class"][name][key],
                    truth["per_class"][name][key], places=12,
                    msg="%s %s" % (name, key))
        for scheme in ("macro", "weighted"):
            for key in ("precision", "recall", "f1-score"):
                self.assertAlmostEqual(
                    pooled["averages"][scheme][key],
                    truth["averages"][scheme][key], places=12,
                    msg="%s %s" % (scheme, key))

    def test_pooling_is_not_the_mean_over_folds(self):
        """The two differ here, which is why the distinction is worth making."""
        report = make_report([self.FOLD_A, self.FOLD_B])
        classes, totals, _ = pooled_counts(report)
        pooled = metrics_from_counts(classes, totals)["averages"]["macro"]["f1-score"]
        fold_mean = sum(f["macro_f1"] for f in report["fold_metrics"]) / 2
        self.assertNotAlmostEqual(pooled, fold_mean, places=4)

    def test_accuracy_is_read_from_the_folds_not_reconstructed(self):
        report = make_report([self.FOLD_A, self.FOLD_B])
        combined = np.asarray(self.FOLD_A) + np.asarray(self.FOLD_B)
        truth = metrics_from_confusion(combined, CLASSES)
        self.assertAlmostEqual(pooled_accuracy(report), truth["accuracy"], places=12)

    def test_a_class_nothing_predicts_does_not_invent_false_positives(self):
        """Precision 0 with TP 0 is exact, not a fallback."""
        matrix = [[0, 0, 100],
                  [0, 50, 0],
                  [0, 0, 900]]
        report = make_report([matrix])
        classes, totals, residual = pooled_counts(report)
        self.assertLess(residual, COUNT_TOLERANCE)
        self.assertEqual(totals[ATTACK]["tp"], 0.0)
        self.assertEqual(totals[ATTACK]["fp"], 0.0)
        self.assertEqual(totals[ATTACK]["fn"], 100.0)

    def test_a_non_integral_true_positive_count_is_fatal(self):
        report = make_report([self.FOLD_A])
        # recall * support must be a count; 0.5 of a row is not.
        report["fold_metrics"][0]["per_class"][ATTACK]["recall"] = 0.505
        report["fold_metrics"][0]["per_class"][ATTACK]["support"] = 101
        with tempfile.TemporaryDirectory() as tmp:
            directory = write_run(tmp, report)
            with self.assertRaises(CrossRunError):
                load_run(directory)

    def test_a_fold_missing_a_declared_class_is_fatal(self):
        report = make_report([self.FOLD_A])
        del report["fold_metrics"][0]["per_class"][OTHER]
        with self.assertRaises(CrossRunError):
            pooled_counts(report)


class PublishedNumbersTests(unittest.TestCase):
    """Against numbers computed the expensive way, from 11M real predictions.

    `run_bootstrap.champion_v2.md` was produced by `bootstrap_run_intervals.py`
    streaming `results/v2-xgboost-none/grouped_predictions.csv`. If the cheap
    reconstruction agrees with it on every class, the cheap path is the same
    measurement rather than a near-enough one.
    """

    RUN = os.path.join(HERE, "..", "..", "results", "v2-xgboost-none")
    EXPECTED = {
        "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE": (0.8828, 0.8298, 0.8555),
        "FULLY_RANDOMIZED_ORIENTEDGRAYHOLE": (0.6529, 0.6634, 0.6581),
        "RANDOMIC_BURST_ORIENTEDGRAYHOLE": (0.7272, 0.8407, 0.7798),
        "RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE": (0.2721, 0.7064, 0.3929),
        "benign_degradation": (0.5768, 0.9134, 0.7071),
        "normal": (0.9992, 0.9863, 0.9927),
    }
    EXPECTED_MACRO_F1 = 0.7310

    def setUp(self):
        if not os.path.exists(os.path.join(self.RUN, "grouped_validation_report.json")):
            self.skipTest("champion run not present in this checkout")

    def test_every_published_per_class_number_is_reproduced(self):
        run = load_run(self.RUN)
        for name, (recall, precision, f1) in self.EXPECTED.items():
            entry = run["metrics"]["per_class"][name]
            self.assertAlmostEqual(entry["recall"], recall, places=4, msg=name)
            self.assertAlmostEqual(entry["precision"], precision, places=4, msg=name)
            self.assertAlmostEqual(entry["f1-score"], f1, places=4, msg=name)

    def test_the_published_macro_f1_is_reproduced(self):
        run = load_run(self.RUN)
        self.assertAlmostEqual(run["metrics"]["averages"]["macro"]["f1-score"],
                               self.EXPECTED_MACRO_F1, places=4)

    def test_the_reconstruction_is_exact_on_the_real_report(self):
        self.assertLess(load_run(self.RUN)["count_residual"], COUNT_TOLERANCE)


class ClassificationTests(unittest.TestCase):
    """A run's card comes from how it was produced, not from where it sits."""

    def test_a_rule_run_is_d2(self):
        self.assertEqual(classify_run({"rule": {"name": "delay"}})[0], "D.2")

    def test_a_tuned_run_is_d4(self):
        self.assertEqual(classify_run({"tuned": "champion-xgboost"})[0], "D.4")

    def test_an_ablated_run_is_d1(self):
        self.assertEqual(classify_run({"feature_set": "no-delta"})[0], "D.1")

    def test_the_reference_feature_set_is_not_an_ablation(self):
        self.assertEqual(classify_run({"feature_set": "all"})[0], "D.3")
        self.assertEqual(classify_run({"feature_set": None})[0], "D.3")
        self.assertEqual(classify_run({})[0], "D.3")

    def test_a_rule_that_designates_one_class_reports_it(self):
        report = {"rule": {"name": "delay"}, "fold_metrics": [
            {"rule": {"designated_attack_class": OTHER}},
            {"rule": {"designated_attack_class": OTHER}},
        ]}
        self.assertEqual(designated_attack_class(report), OTHER)

    def test_a_rule_that_designates_different_classes_per_fold_is_fatal(self):
        """Its per-class row would not be about one class."""
        report = {"rule": {"name": "delay"}, "fold_metrics": [
            {"rule": {"designated_attack_class": OTHER}},
            {"rule": {"designated_attack_class": ATTACK}},
        ]}
        with self.assertRaises(CrossRunError):
            designated_attack_class(report)


class ComparabilityTests(unittest.TestCase):
    """What may not share a table. A mixed table is the dangerous artifact."""

    def _runs(self, *reports):
        runs = []
        for index, report in enumerate(reports):
            classes, totals, residual = pooled_counts(report)
            runs.append({
                "label": "run-%d" % index,
                "report": report,
                "metrics": metrics_from_counts(classes, totals),
                "n_folds": len(report["fold_metrics"]),
                "count_residual": residual,
            })
        return runs

    MATRIX = [[50, 5, 5], [2, 30, 3], [1, 1, 800]]

    def test_two_pools_cannot_share_a_table(self):
        other = make_report([self.MATRIX], dataset_sha256="b" * 64)
        runs = self._runs(make_report([self.MATRIX]), other)
        with self.assertRaises(CrossRunError) as caught:
            check_comparable(runs, allow_smoke=False)
        self.assertIn("dataset", str(caught.exception))

    def test_two_split_files_cannot_share_a_table(self):
        other = make_report([self.MATRIX], splits="/elsewhere/other_splits.json")
        runs = self._runs(make_report([self.MATRIX]), other)
        with self.assertRaises(CrossRunError):
            check_comparable(runs, allow_smoke=False)

    def test_two_protocols_cannot_share_a_table(self):
        other = make_report([self.MATRIX], protocol="group-kfold")
        runs = self._runs(make_report([self.MATRIX]), other)
        with self.assertRaises(CrossRunError):
            check_comparable(runs, allow_smoke=False)

    def test_different_fold_counts_cannot_share_a_table(self):
        runs = self._runs(make_report([self.MATRIX]),
                          make_report([self.MATRIX, self.MATRIX]))
        with self.assertRaises(CrossRunError):
            check_comparable(runs, allow_smoke=False)

    def test_a_technical_smoke_is_refused_by_default(self):
        smoke = make_report([self.MATRIX], status="technical_smoke")
        runs = self._runs(smoke)
        with self.assertRaises(CrossRunError) as caught:
            check_comparable(runs, allow_smoke=False)
        self.assertIn("smoke", str(caught.exception).lower())

    def test_a_technical_smoke_is_allowed_when_asked_for(self):
        smoke = make_report([self.MATRIX], status="technical_smoke")
        check_comparable(self._runs(smoke), allow_smoke=True)

    def test_runs_evaluating_different_row_counts_cannot_share_a_table(self):
        smaller = make_report([[[5, 0, 0], [0, 5, 0], [0, 0, 50]]])
        runs = self._runs(make_report([self.MATRIX]), smaller)
        with self.assertRaises(CrossRunError) as caught:
            check_comparable(runs, allow_smoke=False)
        self.assertIn("row counts", str(caught.exception))

    def test_the_same_configuration_twice_is_fine(self):
        check_comparable(self._runs(make_report([self.MATRIX]),
                                    make_report([self.MATRIX])),
                         allow_smoke=False)


class AveragePrecisionMergeTests(unittest.TestCase):
    """The ranking axis merged in from the curve reports."""

    def _curve_file(self, directory, name, runs):
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"runs": runs}, handle)
        return path

    def _run_entry(self, label, ap, digest=DIGEST):
        return {"label": label, "dataset_sha256": digest,
                "targets": {"ANY_ATTACK": {
                    "average_precision": ap,
                    "average_precision_ci": {"lower": ap - 0.01, "upper": ap + 0.01},
                }}}

    def test_ap_is_merged_from_several_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._curve_file(tmp, "a.json", [self._run_entry("run-a", 0.83)])
            second = self._curve_file(tmp, "b.json", [self._run_entry("run-b", 0.11)])
            found = load_average_precisions([first, second], DIGEST)
            self.assertAlmostEqual(found["run-a"]["average_precision"], 0.83)
            self.assertAlmostEqual(found["run-b"]["average_precision"], 0.11)

    def test_a_curve_file_from_another_pool_is_fatal_not_skipped(self):
        """Silently skipping would leave a withdrawn pool's number readable."""
        with tempfile.TemporaryDirectory() as tmp:
            stale = self._curve_file(
                tmp, "stale.json", [self._run_entry("run-a", 0.9, digest="c" * 64)])
            with self.assertRaises(CrossRunError):
                load_average_precisions([stale], DIGEST)

    def test_two_files_disagreeing_about_one_run_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._curve_file(tmp, "a.json", [self._run_entry("run-a", 0.83)])
            second = self._curve_file(tmp, "b.json", [self._run_entry("run-a", 0.77)])
            with self.assertRaises(CrossRunError) as caught:
                load_average_precisions([first, second], DIGEST)
            self.assertIn("two different", str(caught.exception))

    def test_the_same_run_repeated_with_the_same_ap_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = self._curve_file(tmp, "a.json", [self._run_entry("run-a", 0.83)])
            second = self._curve_file(tmp, "b.json", [self._run_entry("run-a", 0.83)])
            found = load_average_precisions([first, second], DIGEST)
            self.assertAlmostEqual(found["run-a"]["average_precision"], 0.83)


class CliTests(unittest.TestCase):
    MATRIX = [[50, 5, 5], [2, 30, 3], [1, 1, 800]]

    def _table(self, directory):
        with io.open(os.path.join(directory, "out.md"), encoding="utf-8") as fh:
            return fh.read()

    def test_it_writes_a_table_over_several_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, make_report([self.MATRIX]), "alpha")
            write_run(tmp, make_report([self.MATRIX], feature_set="no-delta"), "beta")
            out = os.path.join(tmp, "out.md")
            code = cross_run_main([
                "--runs-glob", os.path.join(tmp, "*"), "--out", out,
                "--json-out", os.path.join(tmp, "out.json"),
            ])
            self.assertEqual(code, 0)
            table = self._table(tmp)
            self.assertIn("`alpha`", table)
            self.assertIn("`beta`", table)
            self.assertIn("This table ranks nothing", table)
            with io.open(os.path.join(tmp, "out.json"), encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertEqual({r["label"] for r in payload["runs"]}, {"alpha", "beta"})
            self.assertEqual({r["card"] for r in payload["runs"]}, {"D.3", "D.1"})

    def test_a_rules_non_designated_classes_are_marked_not_a_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = make_report([self.MATRIX], rule={"name": "delay"})
            for fold in report["fold_metrics"]:
                fold["rule"] = {"designated_attack_class": OTHER}
            write_run(tmp, report, "rule-run")
            out = os.path.join(tmp, "out.md")
            self.assertEqual(cross_run_main(
                ["--runs-glob", os.path.join(tmp, "*"), "--out", out]), 0)
            table = self._table(tmp)
            self.assertIn("n/a", table)
            self.assertIn("not a result", table)

    def test_no_runs_is_an_error_rather_than_an_empty_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cross_run_main(
                ["--runs-glob", os.path.join(tmp, "*"),
                 "--out", os.path.join(tmp, "out.md")]), 1)

    def test_a_mixed_pool_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, make_report([self.MATRIX]), "alpha")
            write_run(tmp, make_report([self.MATRIX], dataset_sha256="b" * 64), "beta")
            self.assertEqual(cross_run_main(
                ["--runs-glob", os.path.join(tmp, "*"),
                 "--out", os.path.join(tmp, "out.md")]), 1)

    def test_confusion_for_a_run_outside_the_table_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, make_report([self.MATRIX]), "alpha")
            self.assertEqual(cross_run_main(
                ["--runs-glob", os.path.join(tmp, "*"),
                 "--confusion", os.path.join(tmp, "nowhere"),
                 "--out", os.path.join(tmp, "out.md")]), 1)


if __name__ == "__main__":
    unittest.main()
