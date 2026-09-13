"""Tests for the run-level bootstrap (checklist D.5/E.4)."""

import json
import os
import tempfile
import unittest

import numpy as np

from bootstrap_run_intervals import (
    MIN_RUNS_FOR_INTERVAL,
    BootstrapError,
    audit_run,
    bootstrap,
    main,
    metrics_from_matrix,
    paired_difference,
    per_run_confusion,
)


CLASSES = ["attack", "normal"]


def write_predictions(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("split_id,row_index,split_group,y_true,y_pred\n")
        for index, (group, truth, pred) in enumerate(rows):
            fh.write("fold-00,%d,%s,%s,%s\n" % (index, group, truth, pred))


def write_run(directory, rows, classes=CLASSES, model="decision-tree"):
    os.makedirs(directory, exist_ok=True)
    write_predictions(os.path.join(directory, "grouped_predictions.csv"), rows)
    with open(os.path.join(directory, "grouped_validation_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"classes": classes, "model": model, "balance": "none"}, fh)


class MetricTests(unittest.TestCase):
    def test_matches_sklearn_on_a_known_matrix(self):
        from sklearn.metrics import precision_recall_fscore_support
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 1, 1, 1])
        matrix = np.bincount(y_true * 2 + y_pred, minlength=4).reshape(2, 2)
        got = metrics_from_matrix(matrix)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], zero_division=0)
        np.testing.assert_allclose(got["precision"], precision)
        np.testing.assert_allclose(got["recall"], recall)
        np.testing.assert_allclose(got["f1"], f1)
        self.assertAlmostEqual(float(got["accuracy"]), 5 / 8)

    def test_absent_class_is_undefined_not_zero(self):
        # No row of class 0 at all: "we have no information" is not the same
        # claim as "recall was zero", and folding a 0 in would bias intervals.
        matrix = np.array([[0, 0], [0, 7]])
        got = metrics_from_matrix(matrix)
        self.assertTrue(np.isnan(got["recall"][0]))
        self.assertTrue(np.isnan(got["precision"][0]))
        self.assertTrue(np.isnan(got["f1"][0]))
        self.assertAlmostEqual(float(got["macro_f1"]), 1.0)  # nanmean over the defined class

    def test_present_but_never_predicted_class_scores_zero_and_stays_in_macro(self):
        # This is the case that matters most on this dataset: the unbalanced
        # models predict no attack row at all. Treating that as "undefined"
        # and dropping it from the macro average is how a detector that finds
        # nothing ends up looking good - the bug this test exists to prevent.
        from sklearn.metrics import classification_report
        y_true = np.array([0] * 3 + [1] * 7)
        y_pred = np.array([1] * 10)                     # class 0 never predicted
        matrix = np.bincount(y_true * 2 + y_pred, minlength=4).reshape(2, 2)
        got = metrics_from_matrix(matrix)
        self.assertEqual(float(got["recall"][0]), 0.0)
        self.assertEqual(float(got["precision"][0]), 0.0)
        self.assertEqual(float(got["f1"][0]), 0.0)
        reference = classification_report(y_true, y_pred, labels=[0, 1],
                                          output_dict=True, zero_division=0)
        self.assertAlmostEqual(float(got["macro_f1"]), reference["macro avg"]["f1-score"])

    def test_macro_f1_matches_sklearn_on_a_six_class_matrix_with_dead_classes(self):
        # The shape of the real `none` runs: four classes present in the data
        # and never predicted, two classes carrying everything.
        from sklearn.metrics import classification_report
        rng = np.random.RandomState(0)
        y_true = np.concatenate([np.full(50, c) for c in range(4)]
                                + [np.full(400, 4), np.full(5000, 5)])
        y_pred = np.where(rng.rand(len(y_true)) < 0.1, 4, 5)
        matrix = np.bincount(y_true * 6 + y_pred, minlength=36).reshape(6, 6)
        got = metrics_from_matrix(matrix)
        reference = classification_report(y_true, y_pred, labels=list(range(6)),
                                          output_dict=True, zero_division=0)
        self.assertAlmostEqual(float(got["macro_f1"]), reference["macro avg"]["f1-score"])

    def test_batched_matrices_give_the_same_answer_as_one_at_a_time(self):
        rng = np.random.RandomState(0)
        batch = rng.randint(0, 50, size=(7, 2, 2))
        batched = metrics_from_matrix(batch)
        for index in range(len(batch)):
            single = metrics_from_matrix(batch[index])
            np.testing.assert_allclose(batched["recall"][index], single["recall"])
            np.testing.assert_allclose(batched["macro_f1"][index], single["macro_f1"])


class PerRunConfusionTests(unittest.TestCase):
    def test_counts_are_split_by_run(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "p.csv")
            write_predictions(path, [
                ("R1", "attack", "attack"), ("R1", "attack", "normal"),
                ("R2", "normal", "normal"), ("R2", "attack", "attack"),
            ])
            groups, counts = per_run_confusion(path, CLASSES, chunk_rows=2)
            self.assertEqual(sorted(groups), ["R1", "R2"])
            self.assertEqual(counts.shape, (2, 2, 2))
            self.assertEqual(int(counts.sum()), 4)
            by_group = {g: counts[i] for i, g in enumerate(groups)}
            self.assertEqual(by_group["R1"].tolist(), [[1, 1], [0, 0]])
            self.assertEqual(by_group["R2"].tolist(), [[1, 0], [0, 1]])

    def test_a_run_spanning_two_chunks_is_not_double_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "p.csv")
            write_predictions(path, [("R1", "attack", "attack")] * 5)
            groups, counts = per_run_confusion(path, CLASSES, chunk_rows=2)
            self.assertEqual(groups, ["R1"])
            self.assertEqual(int(counts[0][0][0]), 5)

    def test_unknown_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = os.path.join(temp, "p.csv")
            write_predictions(path, [("R1", "attack", "something-else")])
            with self.assertRaisesRegex(BootstrapError, "absent from the report"):
                per_run_confusion(path, CLASSES)


class BootstrapTests(unittest.TestCase):
    def test_identical_runs_give_a_degenerate_interval(self):
        # Every replicate sums the same matrix, so there is nothing to vary.
        counts = np.stack([np.array([[4, 1], [0, 5]])] * 12)
        out = bootstrap(counts, iterations=200, seed=1, confidence=0.95)
        np.testing.assert_allclose(out["recall"]["lower"], out["recall"]["upper"])
        np.testing.assert_allclose(out["recall"]["lower"], [0.8, 1.0])

    def test_a_class_concentrated_in_few_runs_gets_a_wider_interval(self):
        # Both classes pool to recall 0.5. `spread` gets there from 20 runs
        # that all agree; `concentrated` from 2 runs that disagree completely.
        # The point of the script is that the second number is far less
        # certain than the first even though they are equal.
        spread, concentrated = 0, 1
        counts = np.zeros((20, 2, 2), dtype=np.int64)
        for index in range(20):
            counts[index, spread, spread] = 5      # 5 correct
            counts[index, spread, 1] = 5           # 5 missed -> recall 0.5
        counts[0, concentrated, concentrated] = 10  # this run: recall 1.0
        counts[1, concentrated, 0] = 10             # that run: recall 0.0
        observed = metrics_from_matrix(counts.sum(axis=0))
        self.assertAlmostEqual(observed["recall"][spread], 0.5)
        self.assertAlmostEqual(observed["recall"][concentrated], 0.5)
        out = bootstrap(counts, iterations=1000, seed=7, confidence=0.95)
        width = out["recall"]["upper"] - out["recall"]["lower"]
        self.assertLess(width[spread], 1e-9)
        self.assertGreater(width[concentrated], 0.5)

    def test_a_single_run_class_collapses_to_zero_width_which_is_why_it_is_flagged(self):
        # Resampling cannot vary a class held by one run: every replicate
        # containing it reproduces its metric exactly, k copies included.
        # A zero-width interval would read as certainty, so `audit_run` marks
        # the class not estimable instead - this test pins the reason.
        counts = np.zeros((20, 2, 2), dtype=np.int64)
        counts[:, 0, 0] = 5
        counts[0, 1, 1] = 100
        counts[0, 1, 0] = 100                       # recall 0.5, from one run
        out = bootstrap(counts, iterations=500, seed=7, confidence=0.95)
        width = out["recall"]["upper"] - out["recall"]["lower"]
        self.assertLess(width[1], 1e-9)
        self.assertEqual(MIN_RUNS_FOR_INTERVAL, 2)

    def test_replicates_without_the_class_are_counted_not_averaged_in(self):
        counts = np.zeros((4, 2, 2), dtype=np.int64)
        counts[:, 1, 1] = 10                       # every run has `normal`
        counts[0, 0, 0] = 3                        # only run 0 has `attack`
        out = bootstrap(counts, iterations=500, seed=3, confidence=0.95)
        undefined = out["recall"]["undefined_replicates"]
        self.assertGreater(int(undefined[0]), 0, "some replicates must miss run 0")
        self.assertEqual(int(undefined[1]), 0)

    def test_confidence_level_widens_the_interval(self):
        rng = np.random.RandomState(0)
        counts = rng.randint(1, 30, size=(15, 2, 2))
        narrow = bootstrap(counts, iterations=800, seed=5, confidence=0.50)
        wide = bootstrap(counts, iterations=800, seed=5, confidence=0.99)
        self.assertLess(narrow["recall"]["upper"][0] - narrow["recall"]["lower"][0],
                        wide["recall"]["upper"][0] - wide["recall"]["lower"][0])


class PairedComparisonTests(unittest.TestCase):
    """The comparison overlapping marginal intervals cannot make."""

    @staticmethod
    def run_fixture(label, counts, groups=None):
        groups = groups or ["R%d" % i for i in range(len(counts))]
        return {"label": label, "groups": groups, "counts": np.asarray(counts)}

    def test_a_consistently_better_model_separates_even_when_margins_overlap(self):
        # Both models vary a lot run to run (so their marginal intervals are
        # wide and overlap), but B beats A on *every* run by the same margin.
        # Paired resampling sees that; comparing two marginal CIs does not.
        rng = np.random.RandomState(0)
        a_counts, b_counts = [], []
        for _ in range(40):
            # Large shared run-to-run difficulty (so both marginals are wide),
            # and a small constant edge for B (so the paired difference is
            # tiny but never negative). That is exactly the regime where
            # comparing marginal intervals gives the wrong answer.
            hard = rng.randint(0, 38)
            a_counts.append([[20 - hard // 2, 20 + hard // 2], [0, 100]])
            b_counts.append([[21 - hard // 2, 19 + hard // 2], [0, 100]])
        first = self.run_fixture("a", a_counts)
        second = self.run_fixture("b", b_counts)
        marginal_a = bootstrap(first["counts"], 600, 1, 0.95)
        marginal_b = bootstrap(second["counts"], 600, 1, 0.95)
        overlap = (marginal_a["macro_f1"]["upper"] >= marginal_b["macro_f1"]["lower"])
        self.assertTrue(overlap, "fixture should have overlapping marginal intervals")
        paired = paired_difference(first, second, "macro_f1", 600, 1, 0.95)
        self.assertGreater(paired["observed_difference"], 0)
        self.assertTrue(paired["separates"])
        self.assertGreater(paired["lower"], 0)

    def test_identical_models_do_not_separate(self):
        counts = np.tile(np.array([[7, 3], [2, 88]]), (25, 1, 1))
        first = self.run_fixture("a", counts)
        second = self.run_fixture("b", counts.copy())
        paired = paired_difference(first, second, "macro_f1", 400, 2, 0.95)
        self.assertAlmostEqual(paired["observed_difference"], 0.0)
        self.assertFalse(paired["separates"])

    def test_runs_are_realigned_by_group_label_not_by_position(self):
        # Same per-run results, but the second run lists its groups in a
        # different order. Pairing by position would fabricate a difference.
        counts = np.array([[[9, 1], [0, 50]], [[2, 8], [0, 50]], [[5, 5], [0, 50]]])
        first = self.run_fixture("a", counts, groups=["R0", "R1", "R2"])
        shuffled = np.array([counts[2], counts[0], counts[1]])
        second = self.run_fixture("b", shuffled, groups=["R2", "R0", "R1"])
        paired = paired_difference(first, second, "macro_f1", 200, 3, 0.95)
        self.assertAlmostEqual(paired["observed_difference"], 0.0)
        self.assertFalse(paired["separates"])

    def test_runs_over_different_run_sets_are_refused(self):
        counts = np.tile(np.array([[7, 3], [2, 88]]), (3, 1, 1))
        first = self.run_fixture("a", counts, groups=["R0", "R1", "R2"])
        second = self.run_fixture("b", counts, groups=["R0", "R1", "R9"])
        with self.assertRaisesRegex(BootstrapError, "not pairable"):
            paired_difference(first, second, "macro_f1", 100, 4, 0.95)


class CliTests(unittest.TestCase):
    def test_end_to_end_writes_a_report_and_flags_thin_classes(self):
        with tempfile.TemporaryDirectory() as temp:
            run = os.path.join(temp, "a-run")
            rows = []
            for index in range(30):
                rows.append(("R%d" % index, "normal", "normal"))
            # 3 runs carry `attack`: enough to resample (>= MIN_RUNS_FOR_INTERVAL)
            # but well under THIN_RUNS, which is the case the warning is for.
            for index in range(3):
                rows.append(("R%d" % index, "attack", "attack"))
            write_run(run, rows)
            out = os.path.join(temp, "boot.md")
            js = os.path.join(temp, "boot.json")
            self.assertEqual(main(["--run", run, "--out", out, "--json-out", js,
                                   "--iterations", "200"]), 0)
            text = open(out, encoding="utf-8").read()
            self.assertIn("Resampling unit: `split_group`", text)
            self.assertNotIn("Paired comparison", text)  # only one run given
            self.assertIn("Thin classes:", text)
            payload = json.load(open(js, encoding="utf-8"))
            self.assertEqual(payload["resampling_unit"], "split_group")
            self.assertEqual(payload["runs"][0]["n_runs"], 30)
            attack = payload["runs"][0]["classes"].index("attack")
            self.assertEqual(payload["runs"][0]["runs_with_class"][attack], 3)
            self.assertTrue(payload["runs"][0]["interval_estimable"][attack])

    def test_single_run_class_is_marked_not_estimable(self):
        with tempfile.TemporaryDirectory() as temp:
            run = os.path.join(temp, "a-run")
            rows = [("R%d" % i, "normal", "normal") for i in range(5)]
            rows += [("R0", "attack", "attack"), ("R0", "attack", "normal")]
            write_run(run, rows)
            audit = audit_run(run, iterations=100, seed=1, confidence=0.95)
            attack = audit["classes"].index("attack")
            normal = audit["classes"].index("normal")
            self.assertEqual(int(audit["runs_with_class"][attack]), 1)
            self.assertFalse(bool(audit["estimable"][attack]))
            self.assertTrue(bool(audit["estimable"][normal]))
            out = os.path.join(temp, "boot.md")
            self.assertEqual(main(["--run", run, "--out", out, "--iterations", "100"]), 0)
            text = open(out, encoding="utf-8").read()
            self.assertIn("not estimable", text)
            self.assertIn("Not estimable:", text)

    def test_bad_arguments_exit_non_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            run = os.path.join(temp, "a-run")
            write_run(run, [("R1", "normal", "normal")])
            self.assertEqual(main(["--run", run, "--confidence", "1.5"]), 1)
            self.assertEqual(main(["--run", run, "--iterations", "0"]), 1)
            self.assertEqual(main(["--run", os.path.join(temp, "missing")]), 1)


if __name__ == "__main__":
    unittest.main()
