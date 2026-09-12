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
