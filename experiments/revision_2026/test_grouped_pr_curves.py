"""Tests for the threshold-curve reporting (`grouped_pr_curves.py`) and for the
posterior persistence it consumes (`run_grouped_validation.py --save-scores`).

Checklist ref.: D.5. What these tests are actually defending:

  - The curve arithmetic matches `sklearn` where `sklearn` has an answer, so
    a reviewer can check AP against a reference implementation rather than
    against this file's own conventions.
  - A threshold is never chosen on the fold it scores. This is the property
    that keeps every recall in the report honest, and it is the one a future
    refactor is most likely to break silently, so it is tested by
    construction: one fold is given a score distribution that would pick a
    wildly different threshold, and the test asserts the fold does *not* get
    that threshold.
  - The prior correction is a no-op on an unbalanced run and undoes a known
    prior shift exactly on a balanced one.
  - `ScoreWriter` round-trips, and `labels_from_proba` agrees with
    `model.predict` - if it ever does not, the persisted scores would
    describe a different classifier than `grouped_predictions.csv` does.
"""

import json
import os
import tempfile
import unittest

import numpy as np

import grouped_pr_curves as curves
from grouped_pr_curves import (
    ANY_ATTACK,
    CurveError,
    effective_prior,
    paired_difference,
    realign,
    target_arrays,
    accumulate_histograms,
    average_precision,
    budget_threshold,
    correction_weights,
    natural_prior,
    operating_point,
    point_metrics,
    pooled_curve,
    suffix_counts,
    threshold_edges,
    threshold_values,
)
from run_grouped_validation import ScoreWriter, labels_from_proba

NORMAL = "normal"
BENIGN = "benign_degradation"
ATTACK_A = "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE"
ATTACK_B = "FULLY_RANDOMIZED_ORIENTEDGRAYHOLE"
CLASSES = [NORMAL, BENIGN, ATTACK_A, ATTACK_B]


def make_report(folds, balance="none", train_counts=None):
    """A minimal `grouped_validation_report.json` payload.

    `folds` maps a split_id to the per-class test support it carried.
    """
    fold_metrics = []
    for split_id, support in folds.items():
        counts = train_counts[split_id] if train_counts else {
            name: max(1, int(support.get(name, 0))) for name in CLASSES
        }
        fold_metrics.append({
            "split_id": split_id,
            "per_class": {name: {"support": float(support.get(name, 0))}
                          for name in CLASSES},
            "balance": {"strategy": balance, "train_class_counts_after": counts},
        })
    return {
        "classes": list(CLASSES),
        "model": "decision-tree",
        "balance": balance,
        "status": "full_grouped_run",
        "dataset_sha256": "0" * 64,
        "scores_file": "grouped_scores.parquet",
        "fold_metrics": fold_metrics,
    }


def write_scores(path, rows, classes=CLASSES):
    """Persist synthetic posteriors through the real ScoreWriter.

    `rows` is a list of (split_id, run_label, true_class, posterior vector).
    Grouped by split_id so each fold becomes its own row group, exactly as a
    real run writes them.
    """
    groups = sorted({row[1] for row in rows})
    group_index = {label: index for index, label in enumerate(groups)}
    class_index = {name: index for index, name in enumerate(classes)}
    writer = ScoreWriter(path, classes, groups)
    try:
        order = []
        for split_id, _, _, _ in rows:
            if split_id not in order:
                order.append(split_id)
        offset = 0
        for split_id in order:
            block = [row for row in rows if row[0] == split_id]
            writer.write_fold(
                split_id,
                np.arange(offset, offset + len(block), dtype="int64"),
                np.asarray([group_index[row[1]] for row in block], dtype="int32"),
                np.asarray([class_index[row[2]] for row in block], dtype="int32"),
                np.asarray([row[3] for row in block], dtype="float32"),
            )
            offset += len(block)
    finally:
        writer.close()


def posterior(attack_a, attack_b=0.0, benign=0.0):
    """A normalised 4-class posterior with `normal` taking the remainder."""
    rest = max(0.0, 1.0 - attack_a - attack_b - benign)
    return [rest, benign, attack_a, attack_b]


class AveragePrecisionTests(unittest.TestCase):
    def test_matches_sklearn_on_a_binary_problem(self):
        """The grid quantisation is the only difference from the reference."""
        from sklearn.metrics import average_precision_score

        rng = np.random.RandomState(0)
        truth = rng.randint(0, 2, size=4000)
        # A score that is informative but far from perfect, so AP lands well
        # inside (prevalence, 1) rather than at a degenerate endpoint.
        score = np.clip(0.5 * truth + rng.normal(0.35, 0.2, size=4000), 1e-6, 1 - 1e-6)

        edges = threshold_edges()
        bins = np.searchsorted(edges, score, side="left")
        counts = np.zeros((2, len(edges) + 1), dtype="int64")
        for label in (0, 1):
            counts[label] = np.bincount(bins[truth == label], minlength=len(edges) + 1)
        cum = suffix_counts(counts)
        ours = average_precision(cum[1].astype(float), cum[0].astype(float),
                                 int((truth == 1).sum()))
        self.assertAlmostEqual(ours, float(average_precision_score(truth, score)),
                               places=3)

    def test_perfect_and_random_separators_hit_their_bounds(self):
        edges = threshold_edges()
        n_bins = len(edges) + 1

        def ap_for(scores, truth):
            bins = np.searchsorted(edges, scores, side="left")
            counts = np.zeros((2, n_bins), dtype="int64")
            for label in (0, 1):
                counts[label] = np.bincount(bins[truth == label], minlength=n_bins)
            cum = suffix_counts(counts)
            return average_precision(cum[1].astype(float), cum[0].astype(float),
                                     int((truth == 1).sum()))

        truth = np.array([0] * 900 + [1] * 100)
        perfect = np.where(truth == 1, 0.99, 0.01)
        self.assertGreater(ap_for(perfect, truth), 0.99)
        constant = np.full(1000, 0.5)
        # A detector with no information scores its own prevalence, which is
        # the floor every AP in the report has to be read against.
        self.assertAlmostEqual(ap_for(constant, truth), 0.1, places=2)

    def test_absent_positives_are_not_estimable_rather_than_zero(self):
        self.assertTrue(np.isnan(average_precision(np.zeros(3), np.zeros(3), 0)))


class BudgetThresholdTests(unittest.TestCase):
    def test_picks_the_loosest_threshold_that_fits(self):
        alerts = np.array([100, 80, 40, 9, 2, 0], dtype="float64")
        # 10% of 100 rows = 10 alerts; index 3 (9 alerts) is the first that fits.
        self.assertEqual(budget_threshold(alerts, 100, 0.10), 3)
        self.assertEqual(budget_threshold(alerts, 100, 1.0), 0)

    def test_unreachable_budget_returns_the_strictest_point(self):
        alerts = np.array([100, 90, 80], dtype="float64")
        self.assertEqual(budget_threshold(alerts, 100, 0.01), 2)

    def test_empty_calibration_set_is_an_error(self):
        with self.assertRaises(CurveError):
            budget_threshold(np.array([1.0]), 0, 0.5)


class PriorCorrectionTests(unittest.TestCase):
    def test_unbalanced_run_weights_are_all_one(self):
        support = {NORMAL: 9000.0, BENIGN: 500.0, ATTACK_A: 300.0, ATTACK_B: 200.0}
        report = make_report({"fold-00": support},
                             train_counts={"fold-00": {k: int(v) for k, v in support.items()}})
        weights = correction_weights(report, "natural")["fold-00"]
        np.testing.assert_allclose(weights, np.ones(len(CLASSES)), rtol=1e-9)

    def test_downsampled_run_weights_recover_the_natural_prior(self):
        support = {NORMAL: 9000.0, BENIGN: 500.0, ATTACK_A: 300.0, ATTACK_B: 200.0}
        balanced = {name: 200 for name in CLASSES}
        report = make_report({"fold-00": support}, balance="downsample",
                             train_counts={"fold-00": balanced})
        weights = correction_weights(report, "natural")["fold-00"]
        expected = natural_prior(report) / 0.25
        np.testing.assert_allclose(weights, expected, rtol=1e-9)
        # A posterior that a balanced model pushed to 50/50 collapses back to
        # roughly the pool's own odds once the training prior is divided out.
        raw = np.array([0.5, 0.0, 0.5, 0.0])
        corrected = raw * weights
        corrected /= corrected.sum()
        self.assertLess(corrected[2], 0.05)

    def test_a_class_absent_from_training_is_refused(self):
        report = make_report({"fold-00": {NORMAL: 10.0}}, balance="downsample",
                             train_counts={"fold-00": {NORMAL: 10, BENIGN: 1,
                                                       ATTACK_A: 0, ATTACK_B: 1}})
        with self.assertRaises(CurveError):
            correction_weights(report, "natural")

    def test_no_correction_requested_returns_none(self):
        report = make_report({"fold-00": {NORMAL: 10.0}})
        self.assertIsNone(correction_weights(report, "as-trained"))


class HistogramTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "grouped_scores.parquet")

    def test_round_trip_preserves_counts_and_recovers_the_prevalence(self):
        rows = []
        for index in range(50):
            rows.append(("fold-00", "run-a", NORMAL, posterior(0.01)))
        for index in range(10):
            rows.append(("fold-00", "run-a", ATTACK_A, posterior(0.90)))
        for index in range(40):
            rows.append(("fold-01", "run-b", NORMAL, posterior(0.02)))
        for index in range(10):
            rows.append(("fold-01", "run-b", ATTACK_B, posterior(0.0, 0.80)))
        write_scores(self.path, rows)

        report = make_report({
            "fold-00": {NORMAL: 50, ATTACK_A: 10},
            "fold-01": {NORMAL: 40, ATTACK_B: 10},
        })
        bundle = accumulate_histograms(self.path, report, None, threshold_edges())
        self.assertEqual(bundle["groups"], ["run-a", "run-b"])
        self.assertEqual(bundle["targets"], [ATTACK_A, ATTACK_B, ANY_ATTACK])

        curve = pooled_curve(bundle, ANY_ATTACK)
        self.assertEqual(curve["rows"], 110)
        self.assertEqual(curve["positives"], 20)
        # Threshold index 0 accepts every row, so the curve must start at
        # perfect recall and the pool's own prevalence.
        self.assertAlmostEqual(curve["tp"][0] / curve["positives"], 1.0)
        self.assertAlmostEqual(curve["tp"][0] / (curve["tp"][0] + curve["fp"][0]),
                               20 / 110)

    def test_each_run_is_pinned_to_the_fold_that_scored_it(self):
        rows = [("fold-00", "run-a", NORMAL, posterior(0.1)),
                ("fold-01", "run-a", NORMAL, posterior(0.1))]
        write_scores(self.path, rows)
        report = make_report({"fold-00": {NORMAL: 1}, "fold-01": {NORMAL: 1}})
        with self.assertRaises(CurveError) as ctx:
            accumulate_histograms(self.path, report, None, threshold_edges())
        self.assertIn("more than one test fold", str(ctx.exception))

    def test_prior_correction_is_applied_per_fold(self):
        rows = [("fold-00", "run-a", NORMAL, posterior(0.5)),
                ("fold-00", "run-a", ATTACK_A, posterior(0.5)),
                ("fold-01", "run-b", NORMAL, posterior(0.5)),
                ("fold-01", "run-b", ATTACK_A, posterior(0.5))]
        write_scores(self.path, rows)
        support = {NORMAL: 900.0, BENIGN: 50.0, ATTACK_A: 30.0, ATTACK_B: 20.0}
        report = make_report({"fold-00": support, "fold-01": support},
                             balance="downsample",
                             train_counts={"fold-00": {n: 100 for n in CLASSES},
                                           "fold-01": {n: 100 for n in CLASSES}})
        edges = threshold_edges()
        weights = correction_weights(report, "natural")
        bundle = accumulate_histograms(self.path, report, weights, edges)
        curve = pooled_curve(bundle, ATTACK_A)
        # Every row scored 0.5 before correction; afterwards all four sit well
        # below it, so a 0.5 threshold now alerts on nothing at all.
        index = int(np.searchsorted(threshold_values(edges), 0.5, side="left"))
        self.assertEqual(int(curve["tp"][index] + curve["fp"][index]), 0)


class OperatingPointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "grouped_scores.parquet")

    def _bundle(self, rows, folds):
        write_scores(self.path, rows)
        return accumulate_histograms(self.path, make_report(folds), None,
                                     threshold_edges())

    def test_cross_fold_threshold_ignores_the_fold_it_scores(self):
        """The property the whole report's honesty rests on.

        `fold-02` is given scores far above every other fold's. If its
        threshold were selected on its own rows it would land up near 0.9;
        selected on the other folds it must land near their much lower
        scores, and alert on essentially all of fold-02.
        """
        rows = []
        for fold, run, level in (("fold-00", "run-a", 0.01),
                                 ("fold-01", "run-b", 0.02)):
            rows += [(fold, run, NORMAL, posterior(level)) for _ in range(99)]
            rows += [(fold, run, ATTACK_A, posterior(level + 0.005))]
        rows += [("fold-02", "run-c", NORMAL, posterior(0.90)) for _ in range(99)]
        rows += [("fold-02", "run-c", ATTACK_A, posterior(0.95))]
        bundle = self._bundle(rows, {
            "fold-00": {NORMAL: 99, ATTACK_A: 1},
            "fold-01": {NORMAL: 99, ATTACK_A: 1},
            "fold-02": {NORMAL: 99, ATTACK_A: 1},
        })

        cross = operating_point(bundle, ATTACK_A, 0.05, "cross-fold")
        thresholds = threshold_values(threshold_edges())
        chosen_for_02 = thresholds[cross["thresholds"]["fold-02"]]
        self.assertLess(chosen_for_02, 0.5)
        run_c = bundle["groups"].index("run-c")
        self.assertEqual(int(cross["tp"][run_c] + cross["fp"][run_c]), 100)

        pooled = operating_point(bundle, ATTACK_A, 0.05, "pooled")
        # Selected on everything at once, the same fold is alerted on far less
        # - which is exactly the optimism cross-fold selection avoids.
        self.assertLess(int(pooled["tp"][run_c] + pooled["fp"][run_c]), 100)

    def test_achieved_alert_rate_tracks_the_budget(self):
        rows = []
        for fold, run in (("fold-00", "run-a"), ("fold-01", "run-b"),
                          ("fold-02", "run-c")):
            for index in range(1000):
                rows.append((fold, run, NORMAL, posterior(index / 2000.0)))
            rows.append((fold, run, ATTACK_A, posterior(0.99)))
        folds = {f: {NORMAL: 1000, ATTACK_A: 1} for f in
                 ("fold-00", "fold-01", "fold-02")}
        bundle = self._bundle(rows, folds)
        loose = point_metrics(operating_point(bundle, ATTACK_A, 0.10, "cross-fold"))
        tight = point_metrics(operating_point(bundle, ATTACK_A, 0.01, "cross-fold"))
        self.assertLessEqual(tight["alert_rate"], loose["alert_rate"])
        self.assertLessEqual(tight["alert_rate"], 0.02)
        self.assertGreaterEqual(tight["precision"], loose["precision"])

    def test_false_alarm_sources_account_for_every_false_positive(self):
        rows = [("fold-00", "run-a", NORMAL, posterior(0.8)),
                ("fold-00", "run-a", BENIGN, posterior(0.8)),
                ("fold-00", "run-a", ATTACK_B, posterior(0.8)),
                ("fold-00", "run-a", ATTACK_A, posterior(0.9)),
                ("fold-01", "run-b", NORMAL, posterior(0.01)),
                ("fold-01", "run-b", ATTACK_A, posterior(0.02))]
        bundle = self._bundle(rows, {
            "fold-00": {NORMAL: 1, BENIGN: 1, ATTACK_A: 1, ATTACK_B: 1},
            "fold-01": {NORMAL: 1, ATTACK_A: 1},
        })
        point = operating_point(bundle, ATTACK_A, 1.0, "cross-fold")
        total = sum(int(source.sum()) for source in point["fp_sources"].values())
        self.assertEqual(total, int(point["fp"].sum()))
        self.assertEqual(set(point["fp_sources"]), {NORMAL, BENIGN, ATTACK_B})


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "grouped_scores.parquet")

    def test_single_run_class_is_reported_as_not_estimable(self):
        rows = []
        for fold, run in (("fold-00", "run-a"), ("fold-01", "run-b")):
            rows += [(fold, run, NORMAL, posterior(0.1)) for _ in range(20)]
        # ATTACK_B exists in exactly one run, so resampling runs gives it no
        # spread at all - a zero-width interval would read as certainty.
        rows += [("fold-00", "run-a", ATTACK_B, posterior(0.0, 0.9))]
        rows += [("fold-00", "run-a", ATTACK_A, posterior(0.9))]
        rows += [("fold-01", "run-b", ATTACK_A, posterior(0.9))]
        write_scores(self.path, rows)
        report = make_report({
            "fold-00": {NORMAL: 20, ATTACK_A: 1, ATTACK_B: 1},
            "fold-01": {NORMAL: 20, ATTACK_A: 1},
        })
        bundle = accumulate_histograms(self.path, report, None, threshold_edges())
        points = [operating_point(bundle, ATTACK_B, 0.5, "cross-fold")]
        result = curves.bootstrap_target(bundle, ATTACK_B, points, 50, 1, 0.95)
        self.assertEqual(result["runs_with_positives"], 1)
        self.assertFalse(result["estimable"])
        self.assertNotIn("ap", result)

        points_a = [operating_point(bundle, ATTACK_A, 0.5, "cross-fold")]
        result_a = curves.bootstrap_target(bundle, ATTACK_A, points_a, 50, 1, 0.95)
        self.assertTrue(result_a["estimable"])
        self.assertLessEqual(result_a["ap"]["lower"], result_a["ap"]["upper"])


class ScorePersistenceTests(unittest.TestCase):
    def test_labels_from_proba_agrees_with_predict(self):
        from sklearn.tree import DecisionTreeClassifier

        rng = np.random.RandomState(3)
        X = rng.normal(size=(400, 5)).astype("float32")
        y = (X[:, 0] > 0).astype("int64") + (X[:, 1] > 0.5).astype("int64")
        model = DecisionTreeClassifier(random_state=0).fit(X, y)
        positions = np.arange(len(X))
        proba = model.predict_proba(X).astype("float32")
        predicted, diagnostics = labels_from_proba(model, X, positions, proba)
        np.testing.assert_array_equal(predicted, model.predict(X))
        self.assertEqual(diagnostics["argmax_mismatches"], 0)
        self.assertFalse(diagnostics["fell_back_to_predict"])
        self.assertEqual(diagnostics["rows_checked"], len(X))

    def test_score_writer_round_trips_through_pyarrow(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "grouped_scores.parquet")
            writer = ScoreWriter(path, CLASSES, ["run-a", "run-b"])
            proba = np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]],
                             dtype="float32")
            writer.write_fold("fold-00", np.array([5, 9], dtype="int64"),
                              np.array([0, 1], dtype="int32"),
                              np.array([0, 2], dtype="int32"), proba)
            writer.close()
            frame = pd.read_parquet(path)
        self.assertEqual(list(frame["row_index"]), [5, 9])
        self.assertEqual([str(v) for v in frame["split_group"]], ["run-a", "run-b"])
        self.assertEqual([str(v) for v in frame["y_true"]], [NORMAL, ATTACK_A])
        np.testing.assert_allclose(frame[["p_%s" % c for c in CLASSES]].to_numpy(),
                                   proba, rtol=1e-6)

    def test_score_writer_rejects_a_mismatched_posterior_block(self):
        from run_grouped_validation import GroupedRunError

        with tempfile.TemporaryDirectory() as tmp:
            writer = ScoreWriter(os.path.join(tmp, "s.parquet"), CLASSES, ["run-a"])
            try:
                with self.assertRaises(GroupedRunError):
                    writer.write_fold("fold-00", np.array([0], dtype="int64"),
                                      np.array([0], dtype="int32"),
                                      np.array([0], dtype="int32"),
                                      np.zeros((2, len(CLASSES)), dtype="float32"))
            finally:
                writer.close()


class EffectivePriorTests(unittest.TestCase):
    def test_auto_corrects_a_balanced_run_and_leaves_an_unbalanced_one(self):
        self.assertEqual(effective_prior({"balance": "downsample"}, "auto"), "natural")
        self.assertEqual(effective_prior({"balance": "smote"}, "auto"), "natural")
        self.assertEqual(effective_prior({"balance": "none"}, "auto"), "as-trained")
        # A report predating the flag records nothing; nothing was rebalanced.
        self.assertEqual(effective_prior({}, "auto"), "as-trained")

    def test_an_explicit_choice_is_never_overridden(self):
        self.assertEqual(effective_prior({"balance": "downsample"}, "as-trained"),
                         "as-trained")
        self.assertEqual(effective_prior({"balance": "none"}, "natural"), "natural")


class PairedComparisonTests(unittest.TestCase):
    """The comparison the marginal intervals cannot make.

    `bootstrap_run_intervals.py` makes this argument for macro F1; these tests
    make it hold for average precision and for budgeted recall, which is what
    D.5's cross-configuration claims rest on.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _analysis(self, name, separation):
        """A run whose attack score separates the classes by `separation`.

        Larger `separation` = a strictly better ranker over identical rows and
        identical runs, which is the ordering the paired test must recover.
        """
        directory = os.path.join(self.tmp.name, name)
        os.makedirs(directory, exist_ok=True)
        rng = np.random.RandomState(11)
        rows, folds = [], {}
        for index in range(6):
            fold = "fold-%02d" % (index % 3)
            run = "run-%d" % index
            for _ in range(200):
                rows.append((fold, run, NORMAL,
                             posterior(float(np.clip(rng.normal(0.2, 0.08), 0.001, 0.99)))))
            for _ in range(20):
                rows.append((fold, run, ATTACK_A,
                             posterior(float(np.clip(rng.normal(0.2 + separation, 0.08),
                                                     0.001, 0.99)))))
            folds.setdefault(fold, {NORMAL: 0, ATTACK_A: 0})
            folds[fold][NORMAL] += 200
            folds[fold][ATTACK_A] += 20
        write_scores(os.path.join(directory, "grouped_scores.parquet"), rows)
        report = make_report(folds)
        with open(os.path.join(directory, "grouped_validation_report.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(report, fh)
        return curves.analyse_run(directory, "auto", "cross-fold", [0.1], 60, 3,
                                  0.95, threshold_edges())

    def test_a_strictly_better_ranker_separates(self):
        weak = self._analysis("weak", 0.10)
        strong = self._analysis("strong", 0.45)
        result = paired_difference(weak, strong, ATTACK_A,
                                   weak["per_target"][ATTACK_A]["points"],
                                   strong["per_target"][ATTACK_A]["points"],
                                   200, 5, 0.95)
        self.assertGreater(result["ap"]["observed"], 0.0)
        self.assertTrue(result["ap"]["separates"])
        self.assertGreater(result["ap"]["lower"], 0.0)

    def test_a_run_paired_against_itself_has_no_difference(self):
        """The null case. A non-zero difference here would mean the pairing is
        resampling the two sides independently instead of together."""
        run = self._analysis("self", 0.30)
        result = paired_difference(run, run, ATTACK_A,
                                   run["per_target"][ATTACK_A]["points"],
                                   run["per_target"][ATTACK_A]["points"],
                                   80, 5, 0.95)
        self.assertEqual(result["ap"]["observed"], 0.0)
        self.assertEqual(result["ap"]["lower"], 0.0)
        self.assertEqual(result["ap"]["upper"], 0.0)
        self.assertFalse(result["ap"]["separates"])
        for block in result["recall"].values():
            self.assertEqual(block["observed"], 0.0)
            self.assertFalse(block["separates"])

    def test_realign_reorders_rather_than_assuming_a_shared_order(self):
        arrays = {"pos_cum": np.array([[1.0], [2.0], [3.0]]),
                  "neg_cum": np.array([[10.0], [20.0], [30.0]]),
                  "pos_per_run": np.array([1.0, 2.0, 3.0]),
                  "groups": ["b", "c", "a"]}
        out = realign(arrays, ["b", "c", "a"], ["a", "b", "c"])
        np.testing.assert_array_equal(out["pos_per_run"], [3.0, 1.0, 2.0])
        self.assertEqual(out["groups"], ["a", "b", "c"])

    def test_runs_covering_different_groups_are_not_pairable(self):
        arrays = {"pos_per_run": np.array([1.0]), "groups": ["a"]}
        with self.assertRaises(CurveError) as ctx:
            realign(arrays, ["a"], ["b"])
        self.assertIn("not pairable", str(ctx.exception))

    def test_target_arrays_reduce_both_runs_the_same_way(self):
        run = self._analysis("shape", 0.30)
        arrays = target_arrays(run["bundle"], ATTACK_A)
        curve = pooled_curve(run["bundle"], ATTACK_A)
        # Summed over runs, the per-run reduction must reproduce the pooled
        # curve exactly - otherwise the paired test and the marginal one would
        # be measuring different quantities.
        np.testing.assert_allclose(arrays["pos_cum"].sum(axis=0), curve["tp"])
        np.testing.assert_allclose(arrays["neg_cum"].sum(axis=0), curve["fp"])
        self.assertEqual(int(arrays["pos_per_run"].sum()), curve["positives"])


class CliTests(unittest.TestCase):
    def test_missing_scores_file_fails_with_a_pointer_to_the_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "grouped_validation_report.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(make_report({"fold-00": {NORMAL: 1, ATTACK_A: 1}}), fh)
            code = curves.main(["--run", tmp, "--out", os.path.join(tmp, "out.md")])
        self.assertEqual(code, 1)

    def test_end_to_end_writes_a_report_and_its_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for fold, run in (("fold-00", "run-a"), ("fold-01", "run-b"),
                              ("fold-02", "run-c")):
                rows += [(fold, run, NORMAL, posterior(0.02)) for _ in range(200)]
                rows += [(fold, run, BENIGN, posterior(0.05)) for _ in range(20)]
                rows += [(fold, run, ATTACK_A, posterior(0.85)) for _ in range(5)]
                rows += [(fold, run, ATTACK_B, posterior(0.0, 0.80)) for _ in range(5)]
            write_scores(os.path.join(tmp, "grouped_scores.parquet"), rows)
            report = make_report({
                f: {NORMAL: 200, BENIGN: 20, ATTACK_A: 5, ATTACK_B: 5}
                for f in ("fold-00", "fold-01", "fold-02")
            })
            with open(os.path.join(tmp, "grouped_validation_report.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(report, fh)

            out = os.path.join(tmp, "pr_curves.md")
            code = curves.main(["--run", tmp, "--out", out, "--iterations", "40",
                                "--budget", "0.05", "--curve-csv",
                                os.path.join(tmp, "curve.csv")])
            self.assertEqual(code, 0)
            with open(out, encoding="utf-8") as fh:
                text = fh.read()
            with open(os.path.splitext(out)[0] + ".json", encoding="utf-8") as fh:
                payload = json.load(fh)

        self.assertIn("Threshold-free ranking", text)
        self.assertNotIn("Paired comparison", text)  # only one run given
        self.assertIn("ANY_ATTACK", text)
        entry = payload["runs"][0]["targets"][ANY_ATTACK]
        self.assertEqual(entry["positives"], 30)
        # Separable by construction, so AP must sit far above the prevalence
        # floor a coin flip would score.
        self.assertGreater(entry["average_precision"], 0.9)
        self.assertLess(entry["prevalence"], 0.15)
        self.assertEqual(len(entry["operating_points"]), 1)
        self.assertEqual(entry["operating_points"][0]["budget"], 0.05)

    def test_rejects_an_impossible_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = curves.main(["--run", tmp, "--budget", "0",
                                "--out", os.path.join(tmp, "out.md")])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
