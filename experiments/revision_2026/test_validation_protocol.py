"""Dependency-light integrity tests for the section B protocol."""

import hashlib
import os
import tempfile
import unittest

import pandas as pd

from generate_grouped_splits import (
    SplitPlanningError,
    _record,
    infer_event_type,
    make_grouped_splits,
    validate_class_coverage,
)
from prepare_grouped_dataset import PreparationError, recompute_trace_deltas
from run_grouped_validation import (
    MODEL_CHOICES,
    GroupedRunError,
    class_counts,
    classifier,
    feature_column_names,
    load_frame,
    load_grouped_arrays,
    prepare_arrays,
    resample_train,
    subsample_train,
    verify_artifacts,
)


class EventTypeTests(unittest.TestCase):
    def test_native_scenario_names(self):
        expected = {
            "SC-DETERMINISTIC_BURST-l100-b5": "SAG.DB",
            "SC-FULLY_RANDOMIZED-l15-b1": "FRG",
            "SC-RANDOMIC_BURST-l15-b5": "SAG.PB",
            "SC-RANDOMIC_MESSAGE-l15-b5": "SAG.PBM",
        }
        self.assertEqual({key: infer_event_type(key) for key in expected}, expected)

    def test_unknown_event_type_fails(self):
        with self.assertRaises(SplitPlanningError):
            infer_event_type("normal-traffic-only")


class ClassCoverageTests(unittest.TestCase):
    def test_closed_set_fold_passes(self):
        labels = {"R1": ["normal", "A"], "R2": ["normal", "A"]}
        rows = {"R1": 10, "R2": 12}
        fold = _record("fold-0", ["R1"], ["R2"], labels, rows)
        self.assertEqual(fold["test_only_labels"], [])
        self.assertEqual(validate_class_coverage([fold]), {})

    def test_unseen_test_class_blocks_standard_evaluation(self):
        labels = {"R1": ["normal", "A"], "R2": ["normal", "B"]}
        rows = {"R1": 10, "R2": 12}
        fold = _record("fold-0", ["R1"], ["R2"], labels, rows)
        with self.assertRaisesRegex(SplitPlanningError, "test labels absent"):
            validate_class_coverage([fold])
        diagnostic = validate_class_coverage([fold], allow_unseen_test_classes=True)
        self.assertEqual(diagnostic, {"fold-0": ["B"]})


class ArtifactBindingTests(unittest.TestCase):
    def test_dataset_is_bound_to_preparation_and_splits_by_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = os.path.join(temp, "prepared.csv")
            with open(dataset, "wb") as fh:
                fh.write(b"split_group,class\nR1,normal\n")
            with open(dataset, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            preparation = {"status": "pass", "output": os.path.abspath(dataset),
                           "output_sha256": digest}
            splits = {"dataset_sha256": digest, "open_set_diagnostic": False}
            self.assertEqual(verify_artifacts(dataset, preparation, splits), digest)

            with open(dataset, "ab") as fh:
                fh.write(b"R2,normal\n")
            with self.assertRaisesRegex(GroupedRunError, "hash differs"):
                verify_artifacts(dataset, preparation, splits)


class DeltaRecomputationTests(unittest.TestCase):
    @staticmethod
    def frame():
        # Deliberately interleaved; source order inside each trace is valid.
        return pd.DataFrame([
            {"trace_id": "A", "GooseTimestamp": 1.0, "StNum": 1, "SqNum": 4,
             "gooseLen": 100, "APDUSize": 50, "frameLen": 120,
             "cbStatus": 0, "t": 1.0, "batch_index": 1},
            {"trace_id": "B", "GooseTimestamp": 3.0, "StNum": 9, "SqNum": 1,
             "gooseLen": 190, "APDUSize": 80, "frameLen": 220,
             "cbStatus": 0, "t": 3.0, "batch_index": 1},
            {"trace_id": "A", "GooseTimestamp": 2.0, "StNum": 2, "SqNum": 0,
             "gooseLen": 110, "APDUSize": 60, "frameLen": 130,
             "cbStatus": 1, "t": 1.5, "batch_index": 1},
            {"trace_id": "B", "GooseTimestamp": 5.0, "StNum": 10, "SqNum": 2,
             "gooseLen": 200, "APDUSize": 90, "frameLen": 230,
             "cbStatus": 0, "t": 4.0, "batch_index": 1},
        ])

    def test_deltas_never_cross_trace_boundaries(self):
        result, audit = recompute_trace_deltas(self.frame())
        self.assertEqual(audit["boundary_rows"], 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["trace_id"].tolist(), ["A", "B"])
        self.assertEqual(result["stDiff"].tolist(), [1.0, 1.0])
        self.assertEqual(result["sqDiff"].tolist(), [-4.0, 1.0])
        self.assertEqual(result["timestampDiff"].tolist(), [1.0, 2.0])
        self.assertEqual(result["cbStatusDiff"].tolist(), [1.0, 0.0])
        self.assertEqual(result["timeFromLastChange"].tolist(), [0.5, 1.0])

    def test_unresolved_trace_is_rejected(self):
        frame = self.frame()
        frame.loc[0, "trace_id"] = "T-UNRESOLVED"
        with self.assertRaisesRegex(PreparationError, "non-independent"):
            recompute_trace_deltas(frame)


class GroupedSplitterTests(unittest.TestCase):
    def test_stratified_group_kfold_keeps_runs_intact(self):
        rows = []
        scenarios = {
            "DB1": "SC-DETERMINISTIC_BURST-l100-b5",
            "DB2": "SC-DETERMINISTIC_BURST-l100-b5",
            "FR1": "SC-FULLY_RANDOMIZED-l15-b1",
            "FR2": "SC-FULLY_RANDOMIZED-l15-b1",
            "PB1": "SC-RANDOMIC_BURST-l15-b5",
            "PB2": "SC-RANDOMIC_BURST-l15-b5",
        }
        attack = {"DB": "DB_ATTACK", "FR": "FR_ATTACK", "PB": "PB_ATTACK"}
        for group, scenario in scenarios.items():
            family = group[:2]
            rows.extend([
                {"split_group": group, "class": "normal", "scenario_id": scenario},
                {"split_group": group, "class": attack[family], "scenario_id": scenario},
            ] * 5)
        frame = pd.DataFrame(rows)
        records, groups, _ = make_grouped_splits(
            frame, "stratified-group-kfold", 2, 42
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(set(groups), set(scenarios))
        for record in records:
            self.assertFalse(set(record["train_groups"]) & set(record["test_groups"]))
            self.assertEqual(record["test_only_labels"], [])

    def test_open_set_splits_are_rejected_by_standard_runner(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = os.path.join(temp, "prepared.csv")
            with open(dataset, "wb") as fh:
                fh.write(b"x")
            digest = hashlib.sha256(b"x").hexdigest()
            preparation = {"status": "pass", "output": os.path.abspath(dataset),
                           "output_sha256": digest}
            splits = {"dataset_sha256": digest, "open_set_diagnostic": True}
            with self.assertRaisesRegex(GroupedRunError, "open-set"):
                verify_artifacts(dataset, preparation, splits)


class BalanceTests(unittest.TestCase):
    """Checklist E: train-only rebalancing (run_grouped_validation.resample_train)."""

    class_names = ["A", "B", "normal"]

    @classmethod
    def frame(cls, counts):
        # counts keyed by class index into class_names, e.g. {0: 20, 1: 8, 2: 100}
        import numpy as np
        rng = np.random.RandomState(0)
        rows = []
        labels = []
        for label, n in counts.items():
            rows.append(rng.normal(size=(n, 2)))
            labels.extend([label] * n)
        X = pd.DataFrame(np.vstack(rows), columns=["f1", "f2"])
        y = np.array(labels)
        return X, y

    def test_none_is_a_no_op(self):
        X, y = self.frame({0: 20, 1: 8, 2: 100})
        X_out, y_out = resample_train(X, y, self.class_names, "none", 0, 20.0, 200_000)
        self.assertIs(X_out, X)
        self.assertIs(y_out, y)

    def test_downsample_equalizes_every_class_to_the_smallest(self):
        X, y = self.frame({0: 20, 1: 8, 2: 100})
        X_out, y_out = resample_train(X, y, self.class_names, "downsample", 0, 20.0, 200_000)
        counts = class_counts(y_out, self.class_names)
        self.assertEqual(counts, {"A": 8, "B": 8, "normal": 8})
        self.assertEqual(len(X_out), 24)

    def test_smote_oversamples_minority_up_to_the_cap_and_leaves_majority_alone(self):
        X, y = self.frame({0: 20, 1: 8, 2: 100})
        X_out, y_out = resample_train(X, y, self.class_names, "smote", 0,
                                       smote_factor=3.0, smote_max_target=50)
        counts = class_counts(y_out, self.class_names)
        # A: 20 * 3 = 60, capped at 50. B: 8 * 3 = 24, under the cap. normal:
        # already >= the 50 cap, so left at its original count untouched.
        self.assertEqual(counts, {"A": 50, "B": 24, "normal": 100})

    def test_smote_rejects_a_class_with_too_few_rows_for_its_neighbourhood(self):
        X, y = self.frame({0: 3, 1: 8, 2: 100})
        with self.assertRaisesRegex(GroupedRunError, "too few"):
            resample_train(X, y, self.class_names, "smote", 0,
                            smote_factor=3.0, smote_max_target=50)


class SubsampleTests(unittest.TestCase):
    """Checklist D.3: the per-fold train cap is a scale control, not balancing."""

    @staticmethod
    def strata(counts):
        import numpy as np
        return np.repeat(np.arange(len(counts)), counts)

    def test_cap_is_a_no_op_when_unset_or_larger_than_the_partition(self):
        strata = self.strata([100, 50])
        self.assertIsNone(subsample_train(strata, 0, 42))
        self.assertIsNone(subsample_train(strata, 150, 42))
        self.assertIsNone(subsample_train(strata, 10_000, 42))

    def test_every_stratum_keeps_its_share_of_the_partition(self):
        import numpy as np
        # One dominant stratum and two small ones: a cap that preserved counts
        # instead of proportions, or that sampled globally at random, would not
        # land on these numbers.
        strata = self.strata([9000, 600, 400])
        keep = subsample_train(strata, 1000, 42)
        kept = np.bincount(strata[keep], minlength=3)
        self.assertEqual(kept.tolist(), [900, 60, 40])
        self.assertEqual(len(keep), 1000)

    def test_a_rare_stratum_is_never_emptied_by_the_cap(self):
        import numpy as np
        # 3 * (100/100_003) rounds down to 0 rows; the floor keeps one, so a
        # cap can never silently remove a rare class from training. That floor
        # is also why the sample may exceed the cap slightly (101 here).
        strata = self.strata([100_000, 3])
        keep = subsample_train(strata, 100, 42)
        kept = np.bincount(strata[keep], minlength=2)
        self.assertEqual(kept.tolist(), [99, 1])
        self.assertEqual(len(keep), 100)

    def test_sample_is_ordered_and_reproducible_for_a_seed(self):
        import numpy as np
        strata = self.strata([500, 500])
        first = subsample_train(strata, 100, 42)
        self.assertTrue(np.all(np.diff(first) > 0), "positions must be sorted and unique")
        np.testing.assert_array_equal(first, subsample_train(strata, 100, 42))
        self.assertFalse(np.array_equal(first, subsample_train(strata, 100, 43)))


class ModelFamilyTests(unittest.TestCase):
    """Checklist D.3: every family is reachable and left at library defaults."""

    def test_every_advertised_family_builds(self):
        for name in MODEL_CHOICES:
            model = classifier(name, seed=7)
            self.assertTrue(hasattr(model, "fit"), name)

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(GroupedRunError, "unknown model"):
            classifier("transformer", seed=7)

    def test_logistic_regression_is_scaled_like_the_baseline_pipeline(self):
        model = classifier("logistic-regression", seed=7)
        self.assertEqual([name for name, _ in model.steps], ["scaler", "clf"])

    def test_no_hyperparameter_is_tuned_away_from_its_default(self):
        from sklearn.ensemble import RandomForestClassifier
        forest = classifier("random-forest", seed=7)
        defaults = RandomForestClassifier()
        for key in ("n_estimators", "max_depth", "max_samples", "min_samples_leaf",
                    "class_weight", "criterion"):
            self.assertEqual(getattr(forest, key), getattr(defaults, key), key)


if __name__ == "__main__":
    unittest.main()

class StreamingLoaderTests(unittest.TestCase):
    """`load_grouped_arrays` must equal the DataFrame path, not approximate it.

    The streaming loader exists because the DataFrame route peaked at 10.02 GB
    on the 265-run pool and stopped fitting on a 15.6 GB machine. A cheaper
    route that silently reordered columns, relabelled classes or rounded
    differently would invalidate every split, metric and SHA-256 binding
    downstream, so equality is asserted element by element rather than argued
    for in a docstring.
    """

    def _fixture(self, directory, row_groups=3, rows_per_group=40):
        """A Parquet file shaped like the real one: many row groups, string
        group/target columns, identifier columns that must be discarded, and
        both integer and floating features."""
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.RandomState(7)
        path = os.path.join(directory, "dataset.parquet")
        writer = None
        try:
            for number in range(row_groups):
                rows = rows_per_group
                frame = pd.DataFrame({
                    # Deliberately not in "nice" order, and the group labels
                    # per row group overlap only partially - which is what
                    # makes the local-to-global remap worth testing.
                    "stNum": rng.randint(0, 5000, rows).astype("int64"),
                    "delta_t": rng.normal(size=rows),
                    "ethDst": ["aa:bb"] * rows,          # BASE_DISCARD_COLUMNS
                    "run_id": ["R%d" % number] * rows,   # IDENTIFIER_COLUMNS
                    "sqNum": rng.randint(0, 100, rows).astype("int64"),
                    "split_group": ["run-%d" % ((number + index) % 4)
                                    for index in range(rows)],
                    "class": [("normal", "benign_degradation", "SAG.DB")[index % 3]
                              for index in range(rows)],
                    "frameLen": rng.uniform(60, 200, rows),
                })
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        return path

    def _dataframe_path(self, path, discard=()):
        """The route `load_grouped_arrays` replaces, reproduced exactly."""
        import pyarrow.parquet as pq
        from run_grouped_validation import BASE_DISCARD_COLUMNS, IDENTIFIER_COLUMNS

        schema_columns = pq.ParquetFile(path).schema_arrow.names
        drop = (IDENTIFIER_COLUMNS | BASE_DISCARD_COLUMNS | set(discard)) - {
            "split_group", "class"}
        columns = [c for c in schema_columns if c not in drop]
        frame = load_frame(path, columns=columns, group_column="split_group",
                           target_column="class")
        return prepare_arrays(frame, "split_group", "class", list(discard))

    def test_matches_the_dataframe_path_element_for_element(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture(tmp)
            expected = self._dataframe_path(path)
            actual = load_grouped_arrays(path, "split_group", "class", [])

        self.assertEqual(actual["features"], expected["features"])
        self.assertEqual(list(actual["classes"]), list(expected["classes"]))
        self.assertEqual(list(actual["group_labels"]), list(expected["group_labels"]))
        np.testing.assert_array_equal(actual["X"], expected["X"])
        self.assertEqual(actual["X"].dtype, expected["X"].dtype)
        np.testing.assert_array_equal(actual["y"], expected["y"])
        self.assertEqual(actual["y"].dtype, expected["y"].dtype)
        np.testing.assert_array_equal(actual["group_codes"], expected["group_codes"])
        np.testing.assert_array_equal(actual["row_index"], expected["row_index"])

    def test_extra_discards_are_honoured_by_both_paths(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture(tmp)
            expected = self._dataframe_path(path, discard=["sqNum"])
            actual = load_grouped_arrays(path, "split_group", "class", ["sqNum"])
        self.assertNotIn("sqNum", actual["features"])
        self.assertEqual(actual["features"], expected["features"])
        np.testing.assert_array_equal(actual["X"], expected["X"])

    def test_feature_column_names_agrees_with_feature_matrix(self):
        """The two paths must reach the same columns from the same sets."""
        from run_grouped_validation import feature_matrix

        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture(tmp)
            frame = pd.read_parquet(path)
            from_frame = list(feature_matrix(frame, "class", []).columns)
        from_names = feature_column_names(list(frame.columns), "class", [])
        self.assertEqual(from_names, from_frame)

    def test_missing_group_column_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._fixture(tmp)
            with self.assertRaises(GroupedRunError):
                load_grouped_arrays(path, "not_a_column", "class", [])

    def test_a_non_numeric_feature_is_refused_before_any_row_is_read(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.parquet")
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({
                "split_group": ["run-0", "run-1"],
                "class": ["normal", "SAG.DB"],
                "a_string_feature": ["x", "y"],
            }), preserve_index=False), path)
            with self.assertRaises(GroupedRunError) as ctx:
                load_grouped_arrays(path, "split_group", "class", [])
        self.assertIn("non-numeric feature columns", str(ctx.exception))

    def test_no_features_left_is_refused(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.parquet")
            pq.write_table(pa.Table.from_pandas(pd.DataFrame({
                "split_group": ["run-0"], "class": ["normal"], "run_id": ["R0"],
            }), preserve_index=False), path)
            with self.assertRaises(GroupedRunError) as ctx:
                load_grouped_arrays(path, "split_group", "class", [])
        self.assertIn("no model features remain", str(ctx.exception))
