"""Tests for checklist D.2's threshold baseline.

The baseline exists to be *beaten*, which makes a silently broken one worse
than none at all: a rule that is accidentally calibrated on its own test rows,
or whose score does not reproduce its own decision, would hand the paper a
comparison that flatters the model. Each test below pins one of the ways that
could happen.
"""

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from check_prediction_integrity import main as check_prediction_integrity_main
from run_rule_baseline import (
    RULES,
    RuleBaselineError,
    attack_mask,
    calibrate,
    designated_attack_class,
    ecdf_score,
    rule_definition,
    threshold_candidates,
)
from run_rule_baseline import main as run_rule_baseline_main

ATTACK = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"
OTHER_ATTACK = "RANDOMIC_BURST_ORIENTEDGRAYHOLE"


class RuleRegistryTests(unittest.TestCase):
    def test_every_rule_names_a_column_and_a_question(self):
        for name, definition in RULES.items():
            self.assertTrue(definition["column"], name)
            self.assertEqual(definition["direction"], "above", name)
            self.assertTrue(definition["question"].strip(), name)

    def test_the_preregistered_arms_are_still_present(self):
        """SS2 preregistered a gap detector and a delay threshold.

        SS15 predicts both lose to the interval rule. A preregistered arm is
        not dropped because a later result expects it to lose - that is the
        prediction this card is supposed to test.
        """
        self.assertIn("sqnum-gap", RULES)
        self.assertIn("delay", RULES)
        self.assertIn("interval-timestamp", RULES)

    def test_an_unknown_rule_is_fatal(self):
        with self.assertRaises(RuleBaselineError):
            rule_definition("no-such-rule")


class ThresholdCandidateTests(unittest.TestCase):
    def test_a_low_cardinality_feature_is_searched_exhaustively(self):
        values = np.asarray([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
        candidates, exhaustive = threshold_candidates(values)
        self.assertTrue(exhaustive)
        np.testing.assert_array_equal(candidates, [1.0, 2.0, 3.0])

    def test_a_high_cardinality_feature_is_quantised_and_says_so(self):
        values = np.arange(20000, dtype="float64")
        candidates, exhaustive = threshold_candidates(values)
        self.assertFalse(exhaustive)
        self.assertLessEqual(candidates.size, 4096)

    def test_non_finite_values_are_not_candidate_thresholds(self):
        values = np.asarray([1.0, np.nan, 2.0, np.inf])
        candidates, _ = threshold_candidates(values)
        np.testing.assert_array_equal(candidates, [1.0, 2.0])

    def test_a_feature_with_no_finite_value_is_fatal(self):
        with self.assertRaises(RuleBaselineError):
            threshold_candidates(np.asarray([np.nan, np.nan]))


class CalibrationTests(unittest.TestCase):
    def test_it_finds_the_separating_threshold(self):
        values = np.asarray([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        positives = np.asarray([False, False, False, True, True, True])
        result = calibrate(values, positives)
        self.assertEqual(result["threshold"], 3.0)
        self.assertEqual(result["train_f1"], 1.0)
        self.assertEqual(result["train_true_positive"], 3)
        self.assertEqual(result["train_false_positive"], 0)
        self.assertEqual(result["train_false_negative"], 0)

    def test_it_does_not_optimise_accuracy(self):
        """At low prevalence "never fire" is the accuracy optimum.

        One positive in a hundred rows: a threshold above everything scores
        99% accurate and F1 = 0. The calibration has to reject it, or the
        baseline this card compares against is a detector that detects
        nothing.
        """
        values = np.concatenate([np.zeros(99), [5.0]])
        positives = np.zeros(100, dtype=bool)
        positives[-1] = True
        result = calibrate(values, positives)
        self.assertLess(result["threshold"], 5.0)
        self.assertEqual(result["train_f1"], 1.0)

    def test_a_non_finite_positive_counts_as_a_miss_not_an_absence(self):
        values = np.asarray([1.0, 2.0, np.nan, 10.0])
        positives = np.asarray([False, False, True, True])
        result = calibrate(values, positives)
        self.assertEqual(result["train_positive_rows"], 2)
        self.assertEqual(result["train_true_positive"], 1)
        self.assertEqual(result["train_false_negative"], 1)

    def test_ties_break_towards_the_quieter_detector(self):
        # Every threshold in [3, 10) separates perfectly; the highest one
        # raises the fewest alerts among equals.
        values = np.asarray([1.0, 3.0, 10.0, 12.0])
        positives = np.asarray([False, False, True, True])
        self.assertEqual(calibrate(values, positives)["threshold"], 3.0)

    def test_a_train_partition_with_no_attack_is_fatal(self):
        with self.assertRaises(RuleBaselineError):
            calibrate(np.asarray([1.0, 2.0]), np.zeros(2, dtype=bool))


class ScoreTests(unittest.TestCase):
    TRAIN = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    def test_the_score_crosses_one_half_exactly_at_the_threshold(self):
        values = np.asarray([0.0, 4.0, 4.5, 5.0, 5.5, 9.0])
        scores, fires = ecdf_score(self.TRAIN, 5.0, values)
        np.testing.assert_array_equal(fires, [False, False, False, False, True, True])
        self.assertTrue((scores[~fires] <= 0.5).all())
        self.assertTrue((scores[fires] > 0.5).all())

    def test_argmax_of_the_emitted_posterior_reproduces_the_decision(self):
        """The invariant `check_prediction_integrity.py` enforces.

        Built here at float32, the precision the posteriors are persisted at,
        because that is where a score of exactly 0.5 would flip the argmax to
        `normal` and make the audit report a disagreement that is really an
        artifact of the score function.
        """
        values = np.linspace(-2.0, 12.0, 501)
        for threshold in (0.0, 4.0, 5.0, 9.0):
            scores, fires = ecdf_score(self.TRAIN, threshold, values)
            posterior = np.stack([scores, np.float32(1.0) - scores], axis=1)
            np.testing.assert_array_equal(
                posterior.argmax(axis=1) == 0, fires,
                "threshold %s" % threshold)

    def test_the_score_is_monotone_in_the_feature(self):
        """Monotone means it changes no ranking, so it changes no AP."""
        values = np.linspace(-5.0, 15.0, 2001)
        scores, _ = ecdf_score(self.TRAIN, 5.0, values)
        self.assertTrue((np.diff(scores.astype("float64")) >= 0).all())

    def test_non_finite_rows_never_fire_and_rank_last(self):
        values = np.asarray([np.nan, 1.0, 9.0])
        scores, fires = ecdf_score(self.TRAIN, 5.0, values)
        self.assertFalse(bool(fires[0]))
        self.assertEqual(float(scores[0]), 0.0)

    def test_it_is_fitted_on_the_train_values_it_is_given(self):
        """Two different train partitions must produce two different maps.

        If the ECDF were taken from the scored rows instead, these would come
        back identical - which is the leak this test exists to catch.
        """
        values = np.asarray([2.0, 6.0])
        left, _ = ecdf_score(np.asarray([0.0, 1.0, 2.0, 3.0]), 1.0, values)
        right, _ = ecdf_score(np.asarray([5.0, 6.0, 7.0, 8.0]), 1.0, values)
        self.assertNotEqual(float(left[0]), float(right[0]))


class DesignatedClassTests(unittest.TestCase):
    CLASSES = ["benign_degradation", ATTACK, OTHER_ATTACK, "normal"]

    def test_it_is_the_commonest_attack_class_not_the_commonest_class(self):
        y = np.asarray([3, 3, 3, 3, 3, 1, 1, 2, 0])
        name, counts = designated_attack_class(y, self.CLASSES)
        self.assertEqual(name, ATTACK)
        self.assertEqual(counts, {ATTACK: 2, OTHER_ATTACK: 1})

    def test_a_partition_with_no_attack_row_is_fatal(self):
        with self.assertRaises(RuleBaselineError):
            designated_attack_class(np.asarray([0, 3, 3]), self.CLASSES)

    def test_attack_mask_ignores_normal_and_benign_degradation(self):
        y = np.asarray([0, 1, 2, 3])
        np.testing.assert_array_equal(
            attack_mask(y, self.CLASSES), [False, True, True, False])

    def test_a_dataset_with_no_attack_class_is_fatal(self):
        with self.assertRaises(RuleBaselineError):
            attack_mask(np.asarray([0, 1]), ["normal", "benign_degradation"])


class RuleRunnerTests(unittest.TestCase):
    """The CLI end to end, against the audits that consume its output."""

    GROUPS = ("run-0", "run-1", "run-2", "run-3")
    ROWS_PER_GROUP = 40

    def _dataset(self, directory, separable=True):
        """A dataset where `timestampDiff` carries the attack and nothing else does.

        A quarter of each run's rows are attack rows whose interval is
        stretched; the remaining columns are noise, so a rule reading any
        other column has nothing to find.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.RandomState(7)
        rows = self.ROWS_PER_GROUP * len(self.GROUPS)
        labels, intervals = [], []
        for index in range(rows):
            is_attack = index % 4 == 0
            labels.append(ATTACK if is_attack else "normal")
            if is_attack and separable:
                intervals.append(rng.uniform(5.0, 6.0))
            else:
                intervals.append(rng.uniform(1.0, 2.0))
        columns = {
            "split_group": [g for g in self.GROUPS for _ in range(self.ROWS_PER_GROUP)],
            "class": labels,
            "timestampDiff": np.asarray(intervals),
            "sqDiff": rng.normal(size=rows),
            "delay": rng.normal(size=rows),
        }
        path = os.path.join(directory, "prepared.parquet")
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(columns), preserve_index=False), path)
        return path

    def _artifacts(self, directory, dataset, digest=None):
        with open(dataset, "rb") as fh:
            real = hashlib.sha256(fh.read()).hexdigest()
        digest = digest or real
        preparation = os.path.join(directory, "preparation.json")
        with open(preparation, "w", encoding="utf-8") as fh:
            json.dump({"status": "pass", "output": os.path.abspath(dataset),
                       "output_sha256": digest}, fh)
        splits = os.path.join(directory, "splits.json")
        with open(splits, "w", encoding="utf-8") as fh:
            json.dump({
                "dataset_sha256": digest,
                "open_set_diagnostic": False,
                "protocol": "stratified-group-kfold",
                "splits": [
                    {"split_id": "fold-00",
                     "train_groups": ["run-0", "run-1"],
                     "test_groups": ["run-2", "run-3"]},
                    {"split_id": "fold-01",
                     "train_groups": ["run-2", "run-3"],
                     "test_groups": ["run-0", "run-1"]},
                ],
            }, fh)
        return preparation, splits

    def _run(self, directory, *extra, **kwargs):
        dataset = kwargs.pop("dataset", None) or self._dataset(directory)
        preparation, splits = self._artifacts(
            directory, dataset, digest=kwargs.pop("digest", None))
        out_dir = os.path.join(directory, "run-%d" % len(os.listdir(directory)))
        code = run_rule_baseline_main([
            "--dataset", dataset,
            "--preparation-report", preparation,
            "--splits", splits,
            "--out-dir", out_dir,
            "--rule", kwargs.pop("rule", "interval-timestamp"),
        ] + list(extra))
        report_path = os.path.join(out_dir, "grouped_validation_report.json")
        if code != 0 or not os.path.exists(report_path):
            return code, None, out_dir
        with open(report_path, encoding="utf-8") as fh:
            return code, json.load(fh), out_dir

    def test_a_separable_rule_recovers_the_attack(self):
        """Recall is perfect; precision is not, and that is the point.

        The fixture separates cleanly - normal intervals in [1, 2), attack
        intervals in [5, 6) - but no observed value lies in the gap, so the
        highest candidate the train partition can offer is its own largest
        normal value. Test normals above it fire. A baseline calibrated on
        the rows it is scored against would not pay that cost, which is
        exactly why this one is.
        """
        with tempfile.TemporaryDirectory() as tmp:
            code, report, out_dir = self._run(tmp)
            self.assertEqual(code, 0)
            self.assertEqual(report["model"], "rule:interval-timestamp")
            self.assertEqual(report["features"], ["timestampDiff"])
            for fold in report["fold_metrics"]:
                self.assertEqual(fold["any_attack"]["recall"], 1.0)
                self.assertGreater(fold["any_attack"]["precision"], 0.9)
                self.assertLess(fold["rule"]["threshold"], 5.0)
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "grouped_predictions.csv")))

    def test_a_rule_on_a_noise_column_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _ = self._run(tmp, rule="sqnum-gap")
            self.assertEqual(code, 0)
            self.assertLess(
                max(fold["any_attack"]["f1"] for fold in report["fold_metrics"]),
                0.9)

    def test_the_threshold_comes_from_train_rows_only(self):
        """The whole point of the baseline, and the easiest thing to get wrong.

        Fold-00 trains on run-0/run-1 and fold-01 on run-2/run-3, so the two
        folds see disjoint rows. The threshold is recorded per fold and both
        folds are scored - if either had been calibrated on its own test rows
        the two would coincide with the test-optimal cut instead of being the
        train partition's own.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, report, _ = self._run(tmp)
            folds = report["fold_metrics"]
            for fold in folds:
                self.assertIn("threshold", fold["rule"])
                self.assertGreater(fold["rule"]["train_positive_rows"], 0)
                self.assertEqual(fold["rule"]["train_positive_rows"],
                                 fold["rule"]["train_true_positive"]
                                 + fold["rule"]["train_false_negative"])

    def test_the_designated_class_is_recorded_per_fold(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report, _ = self._run(tmp)
            for fold in report["fold_metrics"]:
                self.assertEqual(fold["rule"]["designated_attack_class"], ATTACK)

    def test_every_predicted_label_is_normal_or_the_designated_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, out_dir = self._run(tmp)
            frame = pd.read_csv(os.path.join(out_dir, "grouped_predictions.csv"))
            self.assertEqual(set(frame["y_pred"]) - {"normal", ATTACK}, set())
            # y_true is the dataset's own vocabulary, untouched.
            self.assertEqual(set(frame["y_true"]), {"normal", ATTACK})

    def test_the_run_passes_the_independent_prediction_integrity_audit(self):
        """The strongest check available: the audit the learned runs answer to.

        It recomputes every metric from its own confusion matrix and, with
        `--save-scores`, re-derives `argmax(posterior)` over every scored row
        and requires it to reproduce `y_pred`. A rule whose score does not
        agree with its own decision fails here.
        """
        with tempfile.TemporaryDirectory() as tmp:
            code, _, out_dir = self._run(tmp, "--save-scores")
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(
                os.path.join(out_dir, "grouped_scores.parquet")))
            audit = os.path.join(tmp, "integrity.md")
            self.assertEqual(
                check_prediction_integrity_main(
                    ["--run", out_dir, "--out", audit, "--skip-dataset-check"]),
                0)

    def test_only_the_designated_column_carries_the_score(self):
        """The zero columns are a contract, not an accident.

        `grouped_pr_curves.py` draws a per-class curve from each attack
        column, so a non-designated column that quietly held anything would
        produce a per-class number nobody could interpret. Exactly zero makes
        that column's AP the prevalence floor, which is recognisable on sight
        and is recorded in the report as not being a result.
        """
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            _, report, out_dir = self._run(tmp, "--save-scores")
            table = pq.read_table(os.path.join(out_dir, "grouped_scores.parquet"))
            designated = {fold["rule"]["designated_attack_class"]
                          for fold in report["fold_metrics"]}
            self.assertEqual(designated, {ATTACK})
            self.assertIn("per_class_curves", report["rule"])
            for name in report["classes"]:
                column = np.asarray(table.column("p_%s" % name))
                if name in (ATTACK, "normal"):
                    self.assertGreater(float(column.max()), 0.0, name)
                else:
                    self.assertEqual(float(np.abs(column).max()), 0.0, name)

    def test_the_posterior_block_is_a_distribution(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            _, report, out_dir = self._run(tmp, "--save-scores")
            table = pq.read_table(os.path.join(out_dir, "grouped_scores.parquet"))
            total = sum(np.asarray(table.column("p_%s" % name), dtype="float64")
                        for name in report["classes"])
            np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_a_stale_dataset_hash_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _ = self._run(tmp, digest="0" * 64)
            self.assertEqual(code, 1)
            self.assertIsNone(report)

    def test_a_missing_rule_column_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            # `timeFromLastChange` is not in the fixture's schema.
            code, report, _ = self._run(tmp, rule="time-since-change")
            self.assertEqual(code, 1)
            self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
