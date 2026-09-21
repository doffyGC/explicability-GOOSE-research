"""Tests for checklist D.4's nested hyperparameter search.

A tuning run has exactly two ways to be worthless, and both are silent:

  1. **It selects on rows it is then scored on.** The result looks like a
     tuned model beating an untuned one and is really the same leak this
     revision exists to remove, moved one level up. `SelectorIsolationTests`
     pins it structurally - the fits are handed a poisoned array where any
     test row is detectable - rather than by reading the call graph.
  2. **Its "default" point is not actually the default.** Then "tuning bought
     nothing" and "tuning bought something" are both unreadable, because the
     baseline of the comparison is a different model from the D.3 champion.
     `GridRegistryTests` fits both and requires bit-identical posteriors.

The rest cover the selection arithmetic (ties, unscorable points, the two
criteria) and the artifact contract the existing audits consume.
"""

import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from check_prediction_integrity import main as check_prediction_integrity_main
from run_grouped_validation import classifier
from run_nested_tuning import (
    GRIDS,
    GridSelector,
    TuningError,
    attack_columns,
    binding_digest,
    build_estimator,
    expand_grid,
    grid_definition,
    inner_partitions,
    parse_args,
    plan,
    score_point,
    selection_binding,
)
from run_nested_tuning import main as run_nested_tuning_main

def _digest_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


ATTACK = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"
OTHER_ATTACK = "RANDOMIC_BURST_ORIENTEDGRAYHOLE"
CLASSES = np.asarray([ATTACK, OTHER_ATTACK, "normal"], dtype=object)


class GridRegistryTests(unittest.TestCase):
    def test_every_grid_names_its_family_and_its_question(self):
        for name, definition in GRIDS.items():
            self.assertTrue(definition["model"], name)
            self.assertTrue(definition["question"].strip(), name)
            self.assertTrue(definition["axes"], name)

    def test_the_champion_grid_is_twelve_distinct_points(self):
        points = expand_grid("champion-xgboost")
        self.assertEqual(len(points), 12)
        self.assertEqual(len({tuple(sorted(p.items())) for p in points}), 12)

    def test_the_first_point_is_the_library_default_bit_for_bit(self):
        """The baseline of the whole card has to be the D.3 champion itself.

        XGBoost's sklearn wrapper leaves unset parameters as `None` and lets
        the booster apply its own defaults, so "point 0 spells out the
        defaults" is an assumption about the C++ side, not something readable
        from `get_params()`. If it were ever wrong, every D.4 conclusion would
        be a comparison against a model nobody reported.
        """
        rng = np.random.RandomState(3)
        X = rng.normal(size=(600, 6)).astype("float32")
        y = (X[:, 0] > 0).astype("int16")
        y[rng.choice(600, 40, replace=False)] = 2
        default = expand_grid("champion-xgboost")[0]
        tuned = build_estimator("xgboost", default, seed=11, n_jobs=1).fit(X, y)
        champion = classifier("xgboost", 11, n_jobs=1).fit(X, y)
        self.assertTrue(np.array_equal(tuned.predict_proba(X),
                                       champion.predict_proba(X)))

    def test_the_smoke_grid_also_starts_at_the_default(self):
        self.assertEqual(expand_grid("smoke-xgboost")[0], {"max_depth": 6})

    def test_an_unknown_grid_is_fatal(self):
        with self.assertRaises(TuningError):
            grid_definition("no-such-grid")

    def test_a_grid_refuses_a_family_it_was_not_designed_for(self):
        """The axes were read off one family's error structure (SS17).

        Silently accepting `--model random-forest` would produce a run whose
        report names a grid that means nothing on it.
        """
        with tempfile.TemporaryDirectory() as tmp:
            code = run_nested_tuning_main([
                "--dataset", os.path.join(tmp, "absent.parquet"),
                "--preparation-report", os.path.join(tmp, "absent.json"),
                "--splits", os.path.join(tmp, "absent.json"),
                "--out-dir", os.path.join(tmp, "out"),
                "--model", "random-forest", "--plan-only",
            ])
        self.assertEqual(code, 1)

    def test_the_plan_is_a_product_not_a_sum(self):
        self.assertEqual(
            plan(expand_grid("champion-xgboost"), 5, 3),
            {"grid_points": 12, "outer_folds": 5, "inner_splits": 3,
             "inner_fits": 180, "refits": 5, "total_fits": 185})


class ScoreTests(unittest.TestCase):
    def test_any_attack_is_the_sum_of_the_attack_columns(self):
        self.assertEqual(attack_columns(CLASSES), [0, 1])

    def test_a_dataset_with_no_attack_class_is_fatal(self):
        with self.assertRaises(TuningError):
            attack_columns(np.asarray(["normal", "benign_degradation"], dtype=object))

    def test_the_ap_matches_sklearn_on_the_pooled_attack_score(self):
        from sklearn.metrics import average_precision_score

        rng = np.random.RandomState(5)
        proba = rng.dirichlet(np.ones(3), size=400).astype("float32")
        y = rng.randint(0, 3, size=400).astype("int16")
        expected = average_precision_score(
            np.isin(y, [0, 1]).astype("int8"), proba[:, [0, 1]].sum(axis=1))
        self.assertAlmostEqual(score_point(proba, y, CLASSES)["any_attack_ap"],
                               float(expected), places=12)

    def test_a_fold_with_no_attack_row_is_not_ranked_at_zero(self):
        """NaN, not 0.0: an unrankable fold has no opinion about a point.

        Scoring it zero would punish every point identically and drag the
        mean of whichever points happened to meet it.
        """
        proba = np.full((10, 3), 1 / 3, dtype="float32")
        y = np.full(10, 2, dtype="int16")
        result = score_point(proba, y, CLASSES)
        self.assertTrue(np.isnan(result["any_attack_ap"]))
        self.assertEqual(result["positives"], 0)

    def test_macro_f1_is_recorded_beside_the_ap(self):
        proba = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]], dtype="float32")
        result = score_point(proba, np.asarray([0, 2], dtype="int16"), CLASSES)
        self.assertGreater(result["macro_f1"], 0.0)
        self.assertEqual(result["rows"], 2)


class InnerPartitionTests(unittest.TestCase):
    def setUp(self):
        self.groups = np.repeat(np.arange(6, dtype="int32"), 20)
        self.y = np.tile(np.asarray([0, 2, 2, 2] * 5, dtype="int16"), 6)
        self.positions = np.arange(len(self.groups))

    def test_a_group_never_crosses_the_inner_split(self):
        """The same invariant `check_no_leakage.py` enforces outside, inside.

        A hyperparameter chosen on rows from a run that is also in the inner
        train partition is chosen on correlated data - the identical defect
        one level down, and one nothing else in the chain would catch.
        """
        for train_idx, test_idx in inner_partitions(
                self.y, self.groups, self.positions, 3, seed=1):
            train_groups = set(self.groups[self.positions[train_idx]].tolist())
            test_groups = set(self.groups[self.positions[test_idx]].tolist())
            self.assertFalse(train_groups & test_groups)

    def test_every_row_is_tested_exactly_once(self):
        seen = np.concatenate([
            test_idx for _, test_idx in
            inner_partitions(self.y, self.groups, self.positions, 3, seed=1)])
        self.assertEqual(sorted(seen.tolist()), list(range(len(self.positions))))

    def test_too_few_groups_is_fatal_rather_than_a_silent_reduction(self):
        few = self.positions[self.groups < 2]
        with self.assertRaises(TuningError):
            inner_partitions(self.y, self.groups, few, 3, seed=1)


class RecordingEstimator:
    """Records what it was fitted on; returns a posterior it was told to."""

    fitted = []

    def __init__(self, point, proba_for):
        self.point = point
        self.proba_for = proba_for

    def fit(self, X, y):
        RecordingEstimator.fitted.append(np.asarray(X).copy())
        return self

    def predict_proba(self, X):
        return self.proba_for(self.point, np.asarray(X))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class SelectorIsolationTests(unittest.TestCase):
    """What the selector is allowed to see, proven from the arrays it fits."""

    def setUp(self):
        RecordingEstimator.fitted = []
        self.groups = np.repeat(np.arange(8, dtype="int32"), 25)
        rows = len(self.groups)
        rng = np.random.RandomState(9)
        self.X = rng.normal(size=(rows, 3)).astype("float32")
        # The label is a function of column 0, so a synthetic "better point"
        # below can rank it correctly and the selection has a right answer.
        self.boundary = float(np.quantile(self.X[:, 0], 0.75))
        self.y = np.where(self.X[:, 0] > self.boundary, 0, 2).astype("int16")
        # Outer test rows carry a value no train row can hold. Any fit that
        # ever sees it has been handed a test row.
        self.train_positions = np.flatnonzero(self.groups < 6)
        self.test_positions = np.flatnonzero(self.groups >= 6)
        self.X[self.test_positions, 0] = 999.0

    def _selector(self, proba_for, metric="any-attack-ap", points=None):
        import run_nested_tuning

        points = points or [{"max_depth": 6}, {"max_depth": 10}]
        selector = GridSelector(
            "xgboost", points, inner_splits=2, inner_max_rows=0, seed=0,
            n_jobs=1, balance="none", smote_factor=20.0, smote_max_target=100,
            metric=metric)
        original = run_nested_tuning.build_estimator
        run_nested_tuning.build_estimator = (
            lambda model, point, seed, n_jobs: RecordingEstimator(point, proba_for))
        try:
            return selector(fold_index=0, split_id="fold-00", X=self.X, y=self.y,
                            group_codes=self.groups,
                            train_positions=self.train_positions,
                            classes=CLASSES)
        finally:
            run_nested_tuning.build_estimator = original

    def test_no_fit_in_the_search_ever_sees_an_outer_test_row(self):
        self._selector(lambda point, X: np.full((len(X), 3), 1 / 3, dtype="float32"))
        self.assertTrue(RecordingEstimator.fitted)
        for block in RecordingEstimator.fitted:
            self.assertFalse((block == 999.0).any())

    def test_a_tie_keeps_the_default_point(self):
        """Never report a gain that is really a tie.

        Both points score identically here; index 0 is the library default,
        so the conservative reading wins by construction rather than by luck.
        """
        record = self._selector(
            lambda point, X: np.full((len(X), 3), 1 / 3, dtype="float32"))["record"]
        self.assertEqual(record["selected_index"], 0)
        self.assertTrue(record["selected_is_default"])
        self.assertEqual(record["gain_over_default"], 0.0)

    def test_the_better_ranker_is_selected_and_its_gain_recorded(self):
        def proba_for(point, X):
            out = np.full((len(X), 3), 1 / 3, dtype="float32")
            if point["max_depth"] == 10:
                # Ranks the label correctly; the default point does not.
                # The boundary is the one measured before poisoning; taking
                # a quantile here would take it over the 999 sentinels too.
                attack = X[:, 0] > self.boundary
                out[attack] = [0.9, 0.05, 0.05]
            return out
        record = self._selector(proba_for)["record"]
        self.assertEqual(record["selected_index"], 1)
        self.assertFalse(record["selected_is_default"])
        self.assertGreater(record["gain_over_default"], 0.0)

    def test_the_record_carries_both_criteria_for_every_point(self):
        record = self._selector(
            lambda point, X: np.full((len(X), 3), 1 / 3, dtype="float32"))["record"]
        self.assertEqual(len(record["grid"]), 2)
        for row in record["grid"]:
            self.assertIn("mean_any_attack_ap", row)
            self.assertIn("mean_macro_f1", row)
            self.assertEqual(len(row["folds"]), 2)
        self.assertEqual(record["inner"]["groups"], 6)
        self.assertEqual(record["inner"]["rows_available"], len(self.train_positions))

    def test_selecting_on_macro_f1_is_available_and_says_so(self):
        record = self._selector(
            lambda point, X: np.full((len(X), 3), 1 / 3, dtype="float32"),
            metric="macro-f1")["record"]
        self.assertEqual(record["selection_metric"], "macro-f1")
        self.assertEqual(record["score"], record["grid"][0]["mean_macro_f1"])

    def test_a_grid_nothing_can_be_scored_on_is_fatal(self):
        """Better a stopped run than a model chosen by an arbitrary index."""
        self.y[:] = 2  # no attack row anywhere: nothing to rank
        with self.assertRaises(TuningError):
            self._selector(
                lambda point, X: np.full((len(X), 3), 1 / 3, dtype="float32"))

    def test_an_unknown_metric_is_refused_at_construction(self):
        with self.assertRaises(TuningError):
            GridSelector("xgboost", [{"max_depth": 6}], 2, 0, 0, 1, "none",
                         20.0, 100, "accuracy")


BINDING_PLACEHOLDER = "<<binding>>"


class SelectionBindingTests(unittest.TestCase):
    """What the resume cache is keyed on.

    A cached selection is replayed, never re-derived, so the only thing
    standing between a resumed run and a selection it never made is this key.
    Every axis that could move a grid point's score has to be in it - the
    tests are written as "changing X changes the key", so an axis added to
    the runner and forgotten here fails loudly rather than silently widening
    what a cache entry is allowed to answer for.
    """

    SPLIT = {"split_id": "fold-00", "train_groups": ["run-1", "run-0"],
             "test_groups": ["run-2"]}

    def _args(self, *extra):
        return parse_args([
            "--dataset", "d.parquet", "--preparation-report", "p.json",
            "--splits", "s.json", "--out-dir", "out",
        ] + list(extra))

    def _digest(self, args=None, points=None, split=None, dataset_digest="a" * 64,
                fold_index=0):
        return binding_digest(selection_binding(
            args if args is not None else self._args(),
            points or [{"max_depth": 6}],
            split or self.SPLIT, dataset_digest, fold_index))

    def test_the_same_search_is_the_same_key(self):
        self.assertEqual(self._digest(), self._digest())

    def test_group_order_is_not_part_of_the_search(self):
        """Reordering a fold's train groups is not a different search.

        The binding sorts them, so a split file rewritten in another order
        does not throw away a cache that is still valid.
        """
        reordered = dict(self.SPLIT, train_groups=["run-0", "run-1"])
        self.assertEqual(self._digest(), self._digest(split=reordered))

    def test_a_different_dataset_is_a_different_key(self):
        self.assertNotEqual(self._digest(), self._digest(dataset_digest="b" * 64))

    def test_a_different_fold_is_a_different_key(self):
        other = dict(self.SPLIT, split_id="fold-01",
                     train_groups=["run-2", "run-3"])
        self.assertNotEqual(self._digest(), self._digest(split=other))

    def test_the_fold_position_is_part_of_the_key(self):
        """Every fit in the search is seeded at ``seed + fold_index``.

        The same fold read at a different index is a different search, and
        the refit that follows it would be seeded differently too - so the
        index travels with the rest.
        """
        self.assertNotEqual(self._digest(), self._digest(fold_index=1))

    def test_a_different_grid_is_a_different_key(self):
        self.assertNotEqual(
            self._digest(), self._digest(points=[{"max_depth": 10}]))

    def test_every_argument_that_could_move_a_score_is_in_the_key(self):
        for flag, value in [
            ("--grid", "smoke-xgboost"),
            ("--model", "random-forest"),
            ("--selection-metric", "macro-f1"),
            ("--inner-splits", "5"),
            ("--inner-max-rows", "1000"),
            ("--balance", "downsample"),
            ("--smote-oversample-factor", "3.0"),
            ("--smote-max-target", "1000"),
            ("--seed", "7"),
            # A throughput knob rather than a hyperparameter - but XGBoost's
            # histogram reduction order depends on it, and the observed gaps
            # between grid points are ~0.001, so it can flip a selection.
            ("--n-jobs", "2"),
            ("--feature-set", "no-delta"),
            ("--discard-column", "delay"),
            ("--max-train-rows-per-fold", "1000"),
            ("--max-rows-per-group-class", "10"),
        ]:
            with self.subTest(flag=flag):
                self.assertNotEqual(
                    self._digest(), self._digest(args=self._args(flag, value)),
                    "%s does not change the selection key" % flag)

    def test_the_output_directory_is_not_part_of_the_key(self):
        """Where the result is written did not change what was searched.

        Resuming into a fresh output directory is the ordinary case, so it
        must not discard a cache that is still valid.
        """
        self.assertEqual(
            self._digest(),
            self._digest(args=self._args("--out-dir", "somewhere-else")))


class SelectionCacheTests(unittest.TestCase):
    """The resume path: what may be replayed, and what must be recomputed.

    A fold's search is ~38 min on the real pool against ~8 min for the refit
    that follows it, and the D.4 run was killed midway once under memory
    pressure, so replaying a completed search is worth recovering. The risk it
    carries is worse than a slow run - a report naming a selection this
    process never made - so most of what is pinned below is the refusal, and
    the hit is proven by counting fits rather than by reading a log line.
    """

    BINDING = "f" * 64

    def setUp(self):
        RecordingEstimator.fitted = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = os.path.join(self.tmp.name, "selection_cache")
        self.groups = np.repeat(np.arange(8, dtype="int32"), 25)
        rows = len(self.groups)
        rng = np.random.RandomState(9)
        self.X = rng.normal(size=(rows, 3)).astype("float32")
        self.boundary = float(np.quantile(self.X[:, 0], 0.75))
        self.y = np.where(self.X[:, 0] > self.boundary, 0, 2).astype("int16")
        self.train_positions = np.flatnonzero(self.groups < 6)

    def _proba(self, point, X):
        out = np.full((len(X), 3), 1 / 3, dtype="float32")
        if point["max_depth"] == 10:
            out[X[:, 0] > self.boundary] = [0.9, 0.05, 0.05]
        return out

    def _select(self, bindings=None, cache_dir=True):
        import run_nested_tuning

        selector = GridSelector(
            "xgboost", [{"max_depth": 6}, {"max_depth": 10}], inner_splits=2,
            inner_max_rows=0, seed=0, n_jobs=1, balance="none",
            smote_factor=20.0, smote_max_target=100, metric="any-attack-ap",
            cache_dir=self.cache_dir if cache_dir else None,
            bindings={"fold-00": self.BINDING} if bindings is None else bindings)
        original = run_nested_tuning.build_estimator
        run_nested_tuning.build_estimator = (
            lambda model, point, seed, n_jobs: RecordingEstimator(point, self._proba))
        try:
            return selector(fold_index=0, split_id="fold-00", X=self.X, y=self.y,
                            group_codes=self.groups,
                            train_positions=self.train_positions,
                            classes=CLASSES)
        finally:
            run_nested_tuning.build_estimator = original

    def _entry(self):
        return os.path.join(self.cache_dir, "fold-00.json")

    def test_a_completed_search_is_replayed_without_refitting_anything(self):
        first = self._select()["record"]
        self.assertFalse(first["from_cache"])
        self.assertTrue(os.path.exists(self._entry()))

        RecordingEstimator.fitted = []
        second = self._select()["record"]
        self.assertTrue(second["from_cache"])
        # The whole point: the second pass paid for no fit at all.
        self.assertEqual(RecordingEstimator.fitted, [])
        self.assertEqual(second["selected_params"], first["selected_params"])
        self.assertEqual(second["score"], first["score"])
        self.assertEqual(second["grid"], first["grid"])

    def test_the_replayed_estimator_is_the_point_that_was_selected(self):
        """The record is not the result - the refit that follows it is."""
        first = self._select()
        RecordingEstimator.fitted = []
        second = self._select()
        self.assertEqual(second["estimator"].point,
                         first["record"]["selected_params"])
        self.assertEqual(second["estimator"].point, {"max_depth": 10})

    def test_an_entry_written_under_another_binding_is_ignored(self):
        self._select()
        RecordingEstimator.fitted = []
        record = self._select(bindings={"fold-00": "0" * 64})["record"]
        self.assertFalse(record["from_cache"])
        self.assertTrue(RecordingEstimator.fitted)

    def test_a_damaged_entry_is_recomputed_rather_than_trusted(self):
        """A process killed mid-write must not poison the next run.

        Writes are staged and renamed so a truncated entry should not exist,
        but a selection is cheap to recompute and impossible to verify after
        the fact, so anything unreadable is discarded rather than believed.
        """
        damaged = [
            "",
            "{",
            '{"binding_sha256": "%s"}' % BINDING_PLACEHOLDER,
            '{"binding_sha256": "%s", "record": 7}' % BINDING_PLACEHOLDER,
        ]
        for damage in damaged:
            with self.subTest(damage=damage[:24] or "empty"):
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(self._entry(), "w", encoding="utf-8") as fh:
                    fh.write(damage.replace(BINDING_PLACEHOLDER, self.BINDING))
                RecordingEstimator.fitted = []
                record = self._select()["record"]
                self.assertFalse(record["from_cache"])
                self.assertTrue(RecordingEstimator.fitted)

    def test_a_fold_with_no_binding_is_never_read_from_cache(self):
        self._select()
        RecordingEstimator.fitted = []
        record = self._select(bindings={})["record"]
        self.assertFalse(record["from_cache"])
        self.assertTrue(RecordingEstimator.fitted)

    def test_without_a_cache_directory_nothing_is_written_or_read(self):
        """No flag, no cache: the run is what it was before this existed."""
        record = self._select(cache_dir=False)["record"]
        self.assertFalse(record["from_cache"])
        self.assertFalse(os.path.exists(self.cache_dir))


class NestedTuningRunnerTests(unittest.TestCase):
    """The CLI end to end, against the audits that consume its output."""

    GROUPS = ("run-0", "run-1", "run-2", "run-3")
    ROWS_PER_GROUP = 60

    def _dataset(self, directory, seed=4):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rng = np.random.RandomState(seed)
        rows = self.ROWS_PER_GROUP * len(self.GROUPS)
        attack = np.arange(rows) % 4 == 0
        other = np.arange(rows) % 4 == 1
        # Three classes, not two: `classifier("xgboost", ...)` pins
        # `objective="multi:softprob"`, and XGBoost 3.0 refuses that objective
        # on a binary target (`num_class` comes back 0). The real pool has
        # six, so this is the fixture matching the runner, not a workaround.
        labels = np.where(attack, ATTACK, np.where(other, OTHER_ATTACK, "normal"))
        columns = {
            "split_group": [g for g in self.GROUPS for _ in range(self.ROWS_PER_GROUP)],
            "class": labels,
            "timestampDiff": np.where(attack, rng.uniform(5, 6, rows),
                                      np.where(other, rng.uniform(3, 4, rows),
                                               rng.uniform(1, 2, rows))),
            "sqDiff": rng.normal(size=rows),
            "delay": rng.normal(size=rows),
        }
        path = os.path.join(directory, "prepared.parquet")
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame(columns), preserve_index=False), path)
        return path

    def _artifacts(self, directory, dataset, digest=None):
        with open(dataset, "rb") as fh:
            digest = digest or hashlib.sha256(fh.read()).hexdigest()
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
        code = run_nested_tuning_main([
            "--dataset", dataset,
            "--preparation-report", preparation,
            "--splits", splits,
            "--out-dir", out_dir,
            "--grid", kwargs.pop("grid", "smoke-xgboost"),
            "--inner-splits", "2",
            "--inner-max-rows", "0",
        ] + list(extra))
        report_path = os.path.join(out_dir, "grouped_validation_report.json")
        if code != 0 or not os.path.exists(report_path):
            return code, None, out_dir
        with open(report_path, encoding="utf-8") as fh:
            return code, json.load(fh), out_dir

    def test_it_writes_the_same_artifacts_a_plain_run_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, out_dir = self._run(tmp, "--save-scores")
            self.assertEqual(code, 0)
            for name in ("grouped_predictions.csv", "grouped_scores.parquet",
                         "grouped_validation_report.json"):
                self.assertTrue(os.path.exists(os.path.join(out_dir, name)), name)
            self.assertEqual(report["model"], "xgboost")
            self.assertEqual(report["tuned"], "smoke-xgboost")
            self.assertEqual(len(report["fold_metrics"]), 2)

    def test_every_fold_records_how_its_model_was_chosen(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report, _ = self._run(tmp)
            for fold in report["fold_metrics"]:
                selection = fold["selection"]
                self.assertIn(selection["selected_params"], report["tuning"]["points"])
                self.assertEqual(len(selection["grid"]),
                                 len(report["tuning"]["points"]))
                self.assertEqual(selection["inner"]["splits"], 2)
                self.assertLessEqual(selection["inner"]["rows_used"],
                                     selection["inner"]["rows_available"])

    def test_the_report_records_what_was_ruled_out_and_what_was_predicted_null(self):
        """A reader of the result should not have to open the source.

        The exclusions and the expected nulls are the difference between a
        preregistered grid and a grid that was shaped after seeing the
        outcome.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, report, _ = self._run(tmp, grid="champion-xgboost",
                                     dataset=None)
            self.assertIn("class weights / scale_pos_weight",
                          report["tuning"]["excluded_axes"])
            self.assertIn("SAG.PBM", report["tuning"]["expected_nulls"])
            self.assertEqual(report["tuning"]["selection_metric"], "any-attack-ap")

    def test_the_run_passes_the_independent_prediction_integrity_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _, out_dir = self._run(tmp, "--save-scores")
            self.assertEqual(code, 0)
            audit = os.path.join(tmp, "integrity.md")
            self.assertEqual(
                check_prediction_integrity_main(["--run", out_dir, "--out", audit]), 0)

    def test_a_stale_dataset_hash_stops_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, report, _ = self._run(tmp, digest="0" * 64)
            self.assertEqual(code, 1)
            self.assertIsNone(report)

    def test_a_default_run_reports_that_it_searched_every_fold_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report, _ = self._run(tmp)
            cache = report["tuning"]["selection_cache"]
            self.assertIsNone(cache["dir"])
            self.assertEqual(cache["folds_reused"], [])
            for fold in report["fold_metrics"]:
                self.assertFalse(fold["selection"]["from_cache"])

    def test_a_resumed_run_replays_its_selections_and_says_so(self):
        """The CLI end of the resume path, on the artifacts the audits read.

        Two runs over the same dataset, splits and cache: the second must
        reach the same selections without searching for them, write the same
        predictions, and record in its own report which folds it did not
        search - so a reader is never left inferring that 180 fits were paid
        here when they were paid in a process that died.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "selection_cache")
            dataset = self._dataset(tmp)
            first_code, first, first_dir = self._run(
                tmp, "--selection-cache", cache_dir, dataset=dataset)
            self.assertEqual(first_code, 0)
            self.assertEqual(first["tuning"]["selection_cache"]["folds_reused"], [])
            self.assertEqual(
                sorted(os.listdir(cache_dir)), ["fold-00.json", "fold-01.json"])

            second_code, second, second_dir = self._run(
                tmp, "--selection-cache", cache_dir, dataset=dataset)
            self.assertEqual(second_code, 0)
            self.assertEqual(second["tuning"]["selection_cache"]["folds_reused"],
                             ["fold-00", "fold-01"])
            self.assertEqual(
                [fold["selection"]["selected_params"] for fold in second["fold_metrics"]],
                [fold["selection"]["selected_params"] for fold in first["fold_metrics"]])
            # A replay, not a different answer: the refit is seeded and fed
            # exactly as it was, so the predictions have to be the same bytes.
            self.assertEqual(
                _digest_file(os.path.join(second_dir, "grouped_predictions.csv")),
                _digest_file(os.path.join(first_dir, "grouped_predictions.csv")))

    def test_a_cache_written_for_another_dataset_is_not_reused(self):
        """The hash binding reaches the cache too.

        Running the same grid against a different pool is the case where a
        replayed selection would be a fabricated result rather than a
        recovered one.
        """
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = os.path.join(tmp, "selection_cache")
            self._run(tmp, "--selection-cache", cache_dir,
                      dataset=self._dataset(tmp))
            other = os.path.join(tmp, "other")
            os.makedirs(other)
            # A different seed, so this is a different pool rather than a
            # second copy of the same bytes - the fixture is deterministic.
            other_dataset = self._dataset(other, seed=11)
            self.assertNotEqual(_digest_file(other_dataset),
                                _digest_file(os.path.join(tmp, "prepared.parquet")))
            _, report, _ = self._run(tmp, "--selection-cache", cache_dir,
                                     dataset=other_dataset)
            self.assertEqual(report["tuning"]["selection_cache"]["folds_reused"], [])

    def test_plan_only_trains_nothing_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = self._dataset(tmp)
            preparation, splits = self._artifacts(tmp, dataset)
            out_dir = os.path.join(tmp, "planned")
            code = run_nested_tuning_main([
                "--dataset", dataset, "--preparation-report", preparation,
                "--splits", splits, "--out-dir", out_dir, "--plan-only",
            ])
            self.assertEqual(code, 0)
            self.assertFalse(os.path.exists(out_dir))


if __name__ == "__main__":
    unittest.main()
