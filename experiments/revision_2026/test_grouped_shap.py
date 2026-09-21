"""Tests for `run_grouped_shap.py` (card F.3/F.4).

The load-bearing claim of this card is not that SHAP runs - it is that the
model being explained is the model the paper reports. No run persists an
estimator, so the card refits, and a refit that differed in any way would
produce importances for a model nobody published.

The strongest test here is therefore the end-to-end one: produce a reference
run with `run_grouped_validation.py`, then explain it **with verification
on**. If the refit reproduces that run's predictions row for row, then the
fold reconstruction, the seeding and the estimator construction are all
equivalent to the runner's - proven by outcome rather than by reading the two
code paths side by side.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from run_grouped_shap import (
    ShapRunError,
    background_sample,
    load_reference,
    reference_predictions,
    stability,
    stratified_sample,
    verify_fold,
)
from run_grouped_shap import main as shap_main
from run_grouped_validation import main as grouped_main

ATTACK = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"
OTHER = "RANDOMIC_BURST_ORIENTEDGRAYHOLE"
CLASSES = [ATTACK, OTHER, "normal"]


class VerifyFoldTests(unittest.TestCase):
    """The check that makes this card about the published model."""

    ROW_INDEX = np.arange(10, dtype="int64")
    TEST_POSITIONS = np.array([2, 3, 5, 7], dtype="int64")
    PREDICTED = np.array([0, 2, 1, 2], dtype="int16")

    def _reference(self, rows=None, codes=None):
        return {"fold-00": (
            np.asarray(rows if rows is not None else [2, 3, 5, 7], dtype="int64"),
            np.asarray(codes if codes is not None else [0, 2, 1, 2], dtype="int16"),
        )}

    def test_an_identical_refit_passes_and_counts_its_rows(self):
        checked = verify_fold("fold-00", self.ROW_INDEX, self.TEST_POSITIONS,
                              self.PREDICTED, self._reference())
        self.assertEqual(checked, 4)

    def test_it_compares_by_row_index_not_by_position(self):
        """The reference file's order is not part of the claim."""
        checked = verify_fold(
            "fold-00", self.ROW_INDEX, self.TEST_POSITIONS, self.PREDICTED,
            self._reference(rows=[7, 3, 2, 5], codes=[2, 2, 0, 1]))
        self.assertEqual(checked, 4)

    def test_one_disagreeing_row_is_fatal(self):
        with self.assertRaises(ShapRunError) as caught:
            verify_fold("fold-00", self.ROW_INDEX, self.TEST_POSITIONS,
                        self.PREDICTED, self._reference(codes=[0, 2, 1, 0]))
        self.assertIn("different model", str(caught.exception))

    def test_a_different_row_count_is_fatal(self):
        with self.assertRaises(ShapRunError):
            verify_fold("fold-00", self.ROW_INDEX, self.TEST_POSITIONS,
                        self.PREDICTED, self._reference(rows=[2, 3, 5],
                                                        codes=[0, 2, 1]))

    def test_covering_different_rows_is_fatal(self):
        with self.assertRaises(ShapRunError) as caught:
            verify_fold("fold-00", self.ROW_INDEX, self.TEST_POSITIONS,
                        self.PREDICTED, self._reference(rows=[2, 3, 5, 8]))
        self.assertIn("different rows", str(caught.exception))

    def test_a_fold_absent_from_the_reference_is_fatal(self):
        with self.assertRaises(ShapRunError):
            verify_fold("fold-99", self.ROW_INDEX, self.TEST_POSITIONS,
                        self.PREDICTED, self._reference())


class StratifiedSampleTests(unittest.TestCase):
    """The sample preregistered in `explainability_card.md` §3."""

    def setUp(self):
        self.positions = np.arange(600, dtype="int64")
        self.groups = np.repeat(np.arange(6, dtype="int32"), 100)
        self.y = np.tile(np.array([0, 1, 2] * 33 + [2], dtype="int16"), 6)

    def test_an_uncapped_sample_is_the_whole_partition(self):
        sampled = stratified_sample(self.positions, self.groups, self.y, 3, 0, 1)
        self.assertIs(sampled, self.positions)

    def test_a_cap_above_the_partition_changes_nothing(self):
        sampled = stratified_sample(self.positions, self.groups, self.y, 3, 5000, 1)
        self.assertIs(sampled, self.positions)

    def test_every_group_survives_the_cap(self):
        """What shrinks is rows per run, never the set of runs."""
        sampled = stratified_sample(self.positions, self.groups, self.y, 3, 120, 1)
        self.assertLess(len(sampled), len(self.positions))
        self.assertEqual(set(self.groups[sampled].tolist()),
                         set(range(6)))

    def test_every_class_survives_the_cap(self):
        sampled = stratified_sample(self.positions, self.groups, self.y, 3, 120, 1)
        self.assertEqual(set(self.y[sampled].tolist()), {0, 1, 2})

    def test_the_sample_is_reproducible_at_the_same_seed(self):
        first = stratified_sample(self.positions, self.groups, self.y, 3, 120, 7)
        second = stratified_sample(self.positions, self.groups, self.y, 3, 120, 7)
        self.assertTrue(np.array_equal(first, second))


class BackgroundSampleTests(unittest.TestCase):
    """Stratified by class only, and of the size actually asked for.

    Measured, not assumed: stratifying the background by (`split_group`,
    `class`) as well imposes a floor of one row per stratum - a cap of 100
    returned 411 rows on a 265-run smoke, and interventional SHAP costs
    O(rows x background), so the card became four times more expensive than
    the flag said.
    """

    def setUp(self):
        self.positions = np.arange(6000, dtype="int64")
        self.groups = np.repeat(np.arange(200, dtype="int32"), 30)
        rng = np.random.RandomState(3)
        # Imbalanced like the pool: mostly `normal`, a thin attack class.
        self.y = np.where(rng.rand(6000) < 0.02, 0,
                          np.where(rng.rand(6000) < 0.05, 1, 2)).astype("int16")

    def test_it_returns_about_the_size_requested_not_a_stratum_floor(self):
        sample = background_sample(self.positions, self.y, 100, 1)
        self.assertLessEqual(len(sample), 100)
        self.assertGreater(len(sample), 60)

    def test_group_stratification_would_have_overshot(self):
        """The regression this function exists to prevent."""
        grouped = stratified_sample(self.positions, self.groups, self.y, 3, 100, 1)
        self.assertGreater(len(grouped), 3 * len(background_sample(
            self.positions, self.y, 100, 1)))

    def test_no_class_is_lost_however_thin(self):
        """A class absent from the background has a reference the model never sees."""
        sample = background_sample(self.positions, self.y, 100, 1)
        self.assertEqual(set(self.y[sample].tolist()), set(self.y.tolist()))

    def test_an_uncapped_background_is_the_whole_partition(self):
        self.assertIs(background_sample(self.positions, self.y, 0, 1),
                      self.positions)

    def test_a_cap_above_the_partition_changes_nothing(self):
        self.assertIs(background_sample(self.positions, self.y, 99_999, 1),
                      self.positions)

    def test_it_is_reproducible_at_the_same_seed(self):
        first = background_sample(self.positions, self.y, 100, 5)
        second = background_sample(self.positions, self.y, 100, 5)
        self.assertTrue(np.array_equal(first, second))

    def test_it_never_repeats_a_row(self):
        sample = background_sample(self.positions, self.y, 100, 5)
        self.assertEqual(len(sample), len(np.unique(sample)))


class StabilityTests(unittest.TestCase):
    """F.4's spread, reported as a spread rather than as an interval."""

    def test_it_reports_the_folds_it_was_given(self):
        block = stability([0.1, 0.2, 0.15, 0.12, 0.3])
        self.assertEqual(len(block["folds"]), 5)
        self.assertAlmostEqual(block["min"], 0.1)
        self.assertAlmostEqual(block["max"], 0.3)
        self.assertAlmostEqual(block["median"], 0.15)
        self.assertAlmostEqual(block["max_over_median"], 2.0)

    def test_a_feature_with_no_attribution_has_no_ratio(self):
        """Dividing by a zero median would print `inf` as if it meant something."""
        block = stability([0.0, 0.0, 0.0])
        self.assertIsNone(block["max_over_median"])


class ReferenceRunTests(unittest.TestCase):
    """What may be explained, and what may not."""

    def _write(self, directory, **overrides):
        report = {
            "dataset_sha256": "a" * 64,
            "splits": "/somewhere/splits_grouped.json",
            "status": "full_grouped_run",
            "model": "xgboost",
            "seed": 42,
            "n_jobs": -1,
        }
        report.update(overrides)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "grouped_validation_report.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(report, handle)
        return directory

    def test_a_matching_reference_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._write(os.path.join(tmp, "ref"))
            report = load_reference(directory, "a" * 64, "/elsewhere/splits_grouped.json")
            self.assertEqual(report["model"], "xgboost")

    def test_a_reference_from_another_pool_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._write(os.path.join(tmp, "ref"))
            with self.assertRaises(ShapRunError):
                load_reference(directory, "b" * 64, "/x/splits_grouped.json")

    def test_a_reference_on_other_folds_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._write(os.path.join(tmp, "ref"))
            with self.assertRaises(ShapRunError) as caught:
                load_reference(directory, "a" * 64, "/x/other_splits.json")
            self.assertIn("same folds", str(caught.exception))

    def test_a_technical_smoke_may_not_be_explained(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._write(os.path.join(tmp, "ref"),
                                    status="technical_smoke")
            with self.assertRaises(ShapRunError) as caught:
                load_reference(directory, "a" * 64, "/x/splits_grouped.json")
            self.assertIn("wiring check", str(caught.exception))

    def test_missing_predictions_are_reported_rather_than_assumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._write(os.path.join(tmp, "ref"))
            with self.assertRaises(ShapRunError):
                reference_predictions(directory, CLASSES)

    def test_predictions_carrying_an_undeclared_label_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = self._write(os.path.join(tmp, "ref"))
            pd.DataFrame({
                "split_id": ["fold-00"] * 3,
                "row_index": [0, 1, 2],
                "split_group": ["run-0"] * 3,
                "y_true": ["normal"] * 3,
                "y_pred": ["normal", "normal", "SOMETHING_ELSE"],
            }).to_csv(os.path.join(directory, "grouped_predictions.csv"),
                      index=False, encoding="utf-8")
            with self.assertRaises(ShapRunError) as caught:
                reference_predictions(directory, CLASSES)
            self.assertIn("does not declare", str(caught.exception))


class EndToEndTests(unittest.TestCase):
    """Explain a real run, with verification on.

    If this passes, the refit *is* the runner's fit: same fold rows, same
    seed, same estimator. That is the claim card F rests on, established by
    outcome rather than by comparing two code paths by eye.
    """

    GROUPS = ("run-0", "run-1", "run-2", "run-3")
    ROWS_PER_GROUP = 150

    def _dataset(self, directory):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.RandomState(4)
        rows = self.ROWS_PER_GROUP * len(self.GROUPS)
        attack = np.arange(rows) % 4 == 0
        other = np.arange(rows) % 4 == 1
        labels = np.where(attack, ATTACK, np.where(other, OTHER, "normal"))
        frame = pd.DataFrame({
            "split_group": [g for g in self.GROUPS for _ in range(self.ROWS_PER_GROUP)],
            "class": labels,
            "timestampDiff": np.where(attack, rng.uniform(5, 6, rows),
                                      np.where(other, rng.uniform(3, 4, rows),
                                               rng.uniform(1, 2, rows))),
            "sqDiff": rng.normal(size=rows),
            "delay": rng.normal(size=rows),
        })
        path = os.path.join(directory, "prepared.parquet")
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
        return path

    def _artifacts(self, directory, dataset):
        with open(dataset, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        preparation = os.path.join(directory, "preparation.json")
        with open(preparation, "w", encoding="utf-8") as handle:
            json.dump({"status": "pass", "output": os.path.abspath(dataset),
                       "output_sha256": digest}, handle)
        splits = os.path.join(directory, "splits_grouped.json")
        with open(splits, "w", encoding="utf-8") as handle:
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
            }, handle)
        return preparation, splits

    def _reference_run(self, tmp, dataset, preparation, splits):
        out_dir = os.path.join(tmp, "reference")
        code = grouped_main([
            "--dataset", dataset, "--preparation-report", preparation,
            "--splits", splits, "--out-dir", out_dir,
            "--model", "xgboost", "--seed", "42", "--n-jobs", "1",
        ])
        self.assertEqual(code, 0, "reference run failed")
        return out_dir

    def test_the_refit_reproduces_the_reference_run_and_shap_is_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)

            out_dir = os.path.join(tmp, "shap")
            code = shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", out_dir, "--seed", "42", "--n-jobs", "1",
                "--explain-max-rows", "60", "--background-rows", "30",
            ])
            self.assertEqual(code, 0)

            with open(os.path.join(out_dir, "shap_importances.json"),
                      encoding="utf-8") as handle:
                payload = json.load(handle)

            self.assertTrue(payload["verified"])
            self.assertEqual(payload["rows_verified"], 600)
            self.assertEqual(payload["feature_perturbation"], "interventional")
            self.assertEqual(len(payload["folds"]), 2)
            # Per class, never summed across classes.
            self.assertEqual(set(payload["importances"]), set(CLASSES))
            for features in payload["importances"].values():
                self.assertEqual(set(features),
                                 {"timestampDiff", "sqDiff", "delay"})
                for block in features.values():
                    self.assertEqual(len(block["folds"]), 2)

            with open(os.path.join(out_dir, "shap_importances.md"),
                      encoding="utf-8") as handle:
                report = handle.read()
            self.assertIn("the model published", report)
            self.assertNotIn("UNVERIFIED", report)

    def test_the_signal_feature_outranks_the_noise_features(self):
        """A sanity check on the fixture, not a finding about the pool.

        `timestampDiff` is the only column the synthetic label depends on, so
        an explainer that did not rank it first would be misconfigured rather
        than surprising.
        """
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)
            out_dir = os.path.join(tmp, "shap")
            self.assertEqual(shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", out_dir, "--seed", "42", "--n-jobs", "1",
                "--explain-max-rows", "60", "--background-rows", "30",
            ]), 0)
            with open(os.path.join(out_dir, "shap_importances.json"),
                      encoding="utf-8") as handle:
                importances = json.load(handle)["importances"]
            for class_name, features in importances.items():
                ranked = sorted(features.items(), key=lambda kv: -kv[1]["median"])
                self.assertEqual(ranked[0][0], "timestampDiff", class_name)

    def test_a_reference_run_from_another_model_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)
            self.assertEqual(shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", os.path.join(tmp, "shap"),
                "--model", "decision-tree", "--n-jobs", "1",
            ]), 1)

    def test_a_different_seed_is_refused_before_any_fitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)
            out_dir = os.path.join(tmp, "shap")
            self.assertEqual(shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", out_dir, "--seed", "7", "--n-jobs", "1",
            ]), 1)
            self.assertFalse(os.path.exists(
                os.path.join(out_dir, "shap_importances.json")))

    def test_a_different_thread_count_is_refused(self):
        """Not a hyperparameter, but it can move the last decimals."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)
            self.assertEqual(shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", os.path.join(tmp, "shap"), "--n-jobs", "2",
            ]), 1)

    def test_a_capped_smoke_with_verification_on_is_refused_with_a_reason(self):
        """It would fail verification for a reason unrelated to the question."""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)
            self.assertEqual(shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", os.path.join(tmp, "shap"), "--n-jobs", "1",
                "--max-rows-per-group-class", "20",
            ]), 1)

    def test_skipping_verification_marks_the_output_everywhere(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            reference = self._reference_run(tmp, dataset, preparation, splits)
            out_dir = os.path.join(tmp, "shap")
            self.assertEqual(shap_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--reference-run", reference,
                "--out-dir", out_dir, "--n-jobs", "1", "--skip-verification",
                "--explain-max-rows", "60", "--background-rows", "30",
            ]), 0)
            with open(os.path.join(out_dir, "shap_importances.json"),
                      encoding="utf-8") as handle:
                self.assertFalse(json.load(handle)["verified"])
            with open(os.path.join(out_dir, "shap_importances.md"),
                      encoding="utf-8") as handle:
                self.assertIn("UNVERIFIED", handle.read())


if __name__ == "__main__":
    unittest.main()
