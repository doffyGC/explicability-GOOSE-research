"""Threshold curves over the grouped folds - the axis the argmax was hiding.

Checklist ref.: D.5 (per-class reporting), and the question cards D.3/E left
standing.

Why this exists
---------------
Every attack-detection number this revision has published so far is a single
point on a curve nobody measured.  `run_grouped_validation.py` records
``argmax(p)``, so the operating point is fixed by whatever class prior the
*training* partition happened to have:

- `--balance none` trains at the pool's own prior (attacks 0.932%) and
  predicts an attack class for essentially no row.
- `--balance downsample` trains at 1/6 per class, which multiplies the
  attack-vs-rest odds by ``(4/6)/(2/6) / (0.00932/0.99068)`` ~ **213x**.  At a
  fixed argmax every row whose honest attack posterior exceeds ~0.47% is now
  alerted - which is exactly the reported 39-44% false-positive rate on
  *ideal* `normal` traffic, traffic that carries no impairment at all.

Those are not two models disagreeing about whether the attack is detectable.
They are one score read at two thresholds, and reporting them as "recall ~0"
versus "recall 0.70 at a 39% false-positive rate" hides that neither
threshold was ever chosen for anything.  This script recovers the rest of the
curve from the persisted posteriors (`run_grouped_validation.py
--save-scores`) and reports:

1. **A threshold-free number per class** - grid-quantised average precision
   (AP) - so a detector can be judged without picking an operating point at
   all, with the run-level bootstrap `bootstrap_run_intervals.py` argues for.
2. **Operating points at a fixed alert budget**, because that is the
   constraint a substation operator actually has ("how many alerts per 10,000
   messages can anyone act on?"), reported as recall and precision at that
   budget rather than at an arbitrary 0.5.
3. **Where the false alarms come from** at each budget - `normal` versus
   `benign_degradation` versus a different attack class - which is the card-C
   confound measured on the same axis as everything else.

Two things it deliberately does *not* do
----------------------------------------
**A threshold is never chosen on the fold it scores.**  For fold *k* the
threshold is the one meeting the budget on every *other* fold's test
partition.  Folds are group-disjoint, so those rows are a legitimate
calibration set and cost no extra training; picking the threshold on the
scored rows themselves would make every recall in the table optimistic by an
unstated amount.  ``--threshold-selection pooled`` exists to *measure* that
gap, not to report from.

**The bootstrap resamples runs, not rows and not thresholds.**  The per-fold
thresholds are held at their observed values and treated as part of the
model, exactly as a deployed detector would hold them; the interval answers
"if the run matrix had drawn a different set of runs, how much would this
number move?" - the same question, and the same unit, as
`bootstrap_run_intervals.py`.

Usage
-----
    python experiments/revision_2026/grouped_pr_curves.py \\
      --run results/d3-xgboost-none \\
      --out experiments/revision_2026/pr_curves.md

    # a downsampled run's posteriors, corrected back to the deployment prior
    python experiments/revision_2026/grouped_pr_curves.py \\
      --run results/d3-xgboost-downsample --prior natural \\
      --out experiments/revision_2026/pr_curves.downsample.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

from benign_confusion_report import ATTACK_CLASSES, display_name

HERE = os.path.dirname(os.path.abspath(__file__))

ANY_ATTACK = "ANY_ATTACK"
SCORES_FILENAME = "grouped_scores.parquet"

# Threshold grid.  Uniform spacing in probability is useless at both ends of
# this problem: separating a 1e-4 alert rate from a 1e-3 one over 22.7M
# `normal` rows happens entirely inside the last 0.1% of the [0, 1] range.
# Uniform spacing in the *logit* gives constant relative resolution in the
# odds instead - ~2% per step here - and reaches ~2e-9 from either end.
LOGIT_SPAN = 20.0
GRID_EDGES = 2048

# Below this many runs carrying a target's positives there is nothing for a
# run-level bootstrap to resample; mirrors bootstrap_run_intervals.py, where
# the reasoning is written out in full.
MIN_RUNS_FOR_INTERVAL = 2

DEFAULT_BUDGETS = (0.0001, 0.001, 0.01, 0.1)


class CurveError(ValueError):
    pass


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def threshold_edges(span=LOGIT_SPAN, count=GRID_EDGES):
    """Ascending probability grid, uniformly spaced in the logit."""
    import numpy as np

    return 1.0 / (1.0 + np.exp(-np.linspace(-span, span, count)))


def threshold_values(edges):
    """``t[i]`` is the threshold whose accept rule is ``score > t[i]``.

    Index 0 accepts everything, so a curve always starts at recall 1 with the
    pool's own prevalence as its precision - the point every PR curve should
    be read against.
    """
    import numpy as np

    return np.concatenate([[0.0], edges])


def natural_prior(report):
    """The deployment prior: each class's share of the whole evaluated pool.

    Taken from the summed per-fold support rather than from any one fold's
    test partition - the prior a correction targets is a property of the
    dataset being claimed about, not of the fold that happens to be scored.
    """
    import numpy as np

    classes = report["classes"]
    totals = np.zeros(len(classes), dtype="float64")
    for fold in report["fold_metrics"]:
        for index, name in enumerate(classes):
            totals[index] += float(fold["per_class"][name]["support"])
    if totals.sum() <= 0:
        raise CurveError("report records no support, so it has no natural prior")
    return totals / totals.sum()


def train_priors(report):
    """Per fold, the class prior the model was actually fitted at.

    ``train_class_counts_after`` is post-cap and post-balancing, which is the
    prior that shaped the posteriors - the one a correction has to divide out.
    """
    import numpy as np

    classes = report["classes"]
    priors = {}
    for fold in report["fold_metrics"]:
        counts = np.asarray(
            [float(fold["balance"]["train_class_counts_after"][name]) for name in classes],
            dtype="float64")
        if (counts <= 0).any():
            raise CurveError(
                "%s trained without class %s, so its posteriors cannot be "
                "prior-corrected"
                % (fold["split_id"], classes[int((counts <= 0).argmax())]))
        priors[fold["split_id"]] = counts / counts.sum()
    return priors


def correction_weights(report, prior):
    """``pi_k / tau_k`` per fold, or None when no correction was asked for.

    The textbook prior-shift correction: a model fitted at prior ``tau`` and
    deployed at prior ``pi`` has posteriors off by exactly the ratio of the
    two, and ``p_k * pi_k / tau_k``, renormalised, undoes it.  On an
    unbalanced run the weights come out ~1 by construction, which is the
    sanity check that the correction is not inventing a result of its own.
    """
    if prior != "natural":
        return None
    target = natural_prior(report)
    return {split_id: target / tau for split_id, tau in train_priors(report).items()}


def target_names(classes):
    """One one-vs-rest target per attack class present, plus the binary view."""
    attacks = [name for name in classes if name in ATTACK_CLASSES]
    if not attacks:
        raise CurveError("run's classes hold no attack class: %s" % classes)
    return attacks + [ANY_ATTACK], attacks


def scores_for_targets(proba, classes, targets, attacks):
    """The single score each target's curve is drawn over."""
    import numpy as np

    index = {name: position for position, name in enumerate(classes)}
    columns = []
    for name in targets:
        if name == ANY_ATTACK:
            columns.append(proba[:, [index[a] for a in attacks]].sum(axis=1))
        else:
            columns.append(proba[:, index[name]])
    return np.stack(columns, axis=1)


def _codes(series):
    """Integer codes plus their labels, for a column read back as categorical."""
    import numpy as np
    import pandas as pd

    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.cat.codes.to_numpy(dtype="int64"), [str(v) for v in series.cat.categories]
    labels, codes = np.unique(series.to_numpy().astype(str), return_inverse=True)
    return codes.astype("int64"), [str(v) for v in labels]


def _remap(series, index):
    """Codes in a caller-chosen order, whatever order the row group used."""
    import numpy as np

    codes, labels = _codes(series)
    unknown = [label for label in labels if label not in index]
    if unknown:
        raise CurveError("scores hold labels absent from the report: %s" % unknown)
    lookup = np.asarray([index[label] for label in labels], dtype="int64")
    return lookup[codes]


def accumulate_histograms(path, report, weights, edges):
    """Stream the posteriors into per-(target, run, true class, threshold) counts.

    The result is bounded by the grid, not by the dataset: 5 targets x 265
    runs x 6 classes x 2049 thresholds is ~130 MB of int64 whether the pool
    holds 20M rows or 200M.  That is what makes both the curves and a
    1,000-replicate run-level bootstrap cheap afterwards - every later pass
    is a sum over a slice of this array rather than another read of a
    multi-gigabyte scores file.
    """
    import numpy as np
    import pyarrow.parquet as pq

    classes = list(report["classes"])
    targets, attacks = target_names(classes)
    handle = pq.ParquetFile(path)
    proba_columns = ["p_%s" % name for name in classes]
    available = set(handle.schema_arrow.names)
    missing = [c for c in proba_columns + ["split_id", "split_group", "y_true"]
               if c not in available]
    if missing:
        raise CurveError("%s is missing columns: %s" % (path, missing))
    if handle.num_row_groups == 0:
        raise CurveError("%s holds no rows" % path)

    probe = handle.read_row_group(0, columns=["split_group"])
    groups = [str(v) for v in probe.column("split_group").chunk(0).dictionary.to_pylist()]
    del probe
    group_index = {label: position for position, label in enumerate(groups)}
    class_index = {name: position for position, name in enumerate(classes)}

    n_bins = len(edges) + 1
    n_runs, n_classes, n_targets = len(groups), len(classes), len(targets)
    stride_class = n_bins
    stride_run = n_classes * n_bins
    size = n_runs * stride_run
    counts = np.zeros((n_targets, size), dtype="int64")
    run_fold = np.full(n_runs, -1, dtype="int64")
    fold_ids = []

    for number in range(handle.num_row_groups):
        table = handle.read_row_group(
            number, columns=["split_id", "split_group", "y_true"] + proba_columns)
        frame = table.to_pandas()
        del table
        split_codes, split_labels = _codes(frame["split_id"])
        group_codes = _remap(frame["split_group"], group_index)
        class_codes = _remap(frame["y_true"], class_index)
        proba = frame[proba_columns].to_numpy(dtype="float32")
        del frame

        for label in split_labels:
            if label not in fold_ids:
                fold_ids.append(label)
        fold_of = np.asarray([fold_ids.index(label) for label in split_labels],
                             dtype="int64")
        rows_fold = fold_of[split_codes]
        # A run lives in exactly one test fold by construction.  If that ever
        # stopped being true, the cross-fold thresholds below would be applied
        # to rows they were not calibrated away from, silently.
        seen = run_fold[group_codes]
        conflict = (seen >= 0) & (seen != rows_fold)
        if conflict.any():
            raise CurveError("run %s appears in more than one test fold"
                             % groups[int(group_codes[int(conflict.argmax())])])
        run_fold[group_codes] = rows_fold

        if weights is not None:
            unknown = [label for label in split_labels if label not in weights]
            if unknown:
                raise CurveError("no train prior recorded for folds: %s" % unknown)
            per_fold = np.asarray([weights[label] for label in split_labels],
                                  dtype="float32")
            proba = proba * per_fold[split_codes]
            totals = proba.sum(axis=1, keepdims=True)
            np.divide(proba, totals, out=proba, where=totals > 0)

        scores = scores_for_targets(proba, classes, targets, attacks)
        del proba
        base = (group_codes * stride_run) + (class_codes * stride_class)
        for position in range(n_targets):
            bins = np.searchsorted(edges, scores[:, position], side="left")
            counts[position] += np.bincount(base + bins, minlength=size)
        del scores, base, group_codes, class_codes

    if (run_fold < 0).any():
        blank = [groups[i] for i in np.flatnonzero(run_fold < 0)]
        raise CurveError("no scored rows for runs: %s" % blank[:5])
    return {
        "counts": counts.reshape(n_targets, n_runs, n_classes, n_bins),
        "groups": groups, "classes": classes, "targets": targets,
        "attacks": attacks, "run_fold": run_fold, "folds": fold_ids,
    }


def suffix_counts(counts):
    """``out[..., i]`` = rows whose score exceeds threshold ``i``."""
    return counts[..., ::-1].cumsum(axis=-1)[..., ::-1]


def target_partition(bundle, target):
    """Which true classes count as positives for this target, and which do not."""
    classes = bundle["classes"]
    if target == ANY_ATTACK:
        positives = [classes.index(name) for name in bundle["attacks"]]
    else:
        positives = [classes.index(target)]
    negatives = [index for index in range(len(classes)) if index not in positives]
    return positives, negatives


def average_precision(tp, fp, positives):
    """Grid-quantised average precision, following scikit-learn's step rule.

    ``AP = sum_i (R_i - R_{i+1}) * P_i`` walking the grid from the loosest
    threshold to the strictest, plus the strictest point's own contribution -
    the same "no linear interpolation between operating points" convention
    ``sklearn.metrics.average_precision_score`` uses.  Quantised, because the
    thresholds are a fixed grid rather than every distinct score in the data;
    with 2048 logit-spaced steps the difference is far below the run-level
    interval reported next to it.
    """
    import numpy as np

    if positives <= 0:
        return float("nan")
    recall = tp / float(positives)
    alerted = tp + fp
    precision = np.divide(tp, alerted, out=np.zeros_like(tp, dtype="float64"),
                          where=alerted > 0)
    steps = recall[:-1] - recall[1:]
    return float((steps * precision[:-1]).sum() + recall[-1] * precision[-1])


def budget_threshold(alerts, rows, budget):
    """Strictest-but-sufficient threshold index meeting an alert budget.

    ``alerts`` is monotonically non-increasing in the index, so the smallest
    index meeting the budget is the one with the most recall among those that
    fit.  When even the strictest grid point overshoots, the strictest point
    is returned and the caller reports the achieved rate, which will be above
    the budget - an honest "this model cannot be run this quietly" rather
    than a silently relaxed constraint.
    """
    import numpy as np

    if rows <= 0:
        raise CurveError("cannot pick a threshold from an empty calibration set")
    rate = alerts / float(rows)
    feasible = np.flatnonzero(rate <= budget)
    return int(feasible[0]) if feasible.size else int(len(rate) - 1)


def operating_point(bundle, target, budget, selection):
    """Per-run counts at the budget's threshold, with the threshold itself.

    Everything the pooled table and the bootstrap need is reduced to one
    number per run here, so a replicate is a sum over 265 values rather than
    another pass over the grid.
    """
    import numpy as np

    positives, negatives = target_partition(bundle, target)
    index = bundle["targets"].index(target)
    per_run = bundle["counts"][index]                      # (run, class, bin)
    cum = suffix_counts(per_run)                           # (run, class, bin)
    pos_cum = cum[:, positives, :].sum(axis=1)
    neg_cum = cum[:, negatives, :].sum(axis=1)
    rows_per_run = per_run.sum(axis=(1, 2))
    pos_per_run = per_run[:, positives, :].sum(axis=(1, 2))
    run_fold = bundle["run_fold"]

    chosen = np.zeros(len(rows_per_run), dtype="int64")
    thresholds = {}
    if selection == "pooled":
        pick = budget_threshold((pos_cum + neg_cum).sum(axis=0),
                                rows_per_run.sum(), budget)
        chosen[:] = pick
        thresholds["pooled"] = pick
    else:
        for fold in range(len(bundle["folds"])):
            other = run_fold != fold
            if not other.any():
                raise CurveError("cross-fold selection needs at least two folds")
            pick = budget_threshold((pos_cum[other] + neg_cum[other]).sum(axis=0),
                                    int(rows_per_run[other].sum()), budget)
            chosen[run_fold == fold] = pick
            thresholds[bundle["folds"][fold]] = pick

    rows = np.arange(len(chosen))
    tp = pos_cum[rows, chosen]
    fp = neg_cum[rows, chosen]
    sources = {
        bundle["classes"][cls]: cum[rows, cls, chosen]
        for cls in negatives
    }
    return {
        "target": target, "budget": budget, "selection": selection,
        "threshold_index": chosen, "thresholds": thresholds,
        "tp": tp, "fp": fp, "positives": pos_per_run, "rows": rows_per_run,
        "fp_sources": sources,
    }


def point_metrics(point, picked=None):
    """Recall, precision and alert rate from a (possibly resampled) set of runs."""
    import numpy as np

    take = slice(None) if picked is None else picked
    tp = float(point["tp"][take].sum())
    fp = float(point["fp"][take].sum())
    positives = float(point["positives"][take].sum())
    rows = float(point["rows"][take].sum())
    return {
        "recall": tp / positives if positives > 0 else float("nan"),
        "precision": (tp / (tp + fp)) if (tp + fp) > 0 else float("nan"),
        "alert_rate": (tp + fp) / rows if rows > 0 else float("nan"),
        "tp": tp, "fp": fp, "positives": positives, "rows": rows,
    }


def pooled_curve(bundle, target):
    """The whole threshold sweep, pooled over every run."""
    import numpy as np

    positives, negatives = target_partition(bundle, target)
    index = bundle["targets"].index(target)
    totals = bundle["counts"][index].sum(axis=0)           # (class, bin)
    cum = suffix_counts(totals)
    tp = cum[positives, :].sum(axis=0).astype("float64")
    fp = cum[negatives, :].sum(axis=0).astype("float64")
    n_positives = int(totals[positives, :].sum())
    rows = int(totals.sum())
    return {"tp": tp, "fp": fp, "positives": n_positives, "rows": rows,
            "ap": average_precision(tp, fp, n_positives),
            "prevalence": n_positives / rows if rows else float("nan")}


def target_arrays(bundle, target):
    """Per-run cumulative positive/negative counts for one target.

    Factored out because three callers need exactly this - the marginal
    bootstrap, the paired bootstrap and the operating points - and because a
    paired comparison is only meaningful if both runs are reduced the same
    way.
    """
    import numpy as np

    positives, negatives = target_partition(bundle, target)
    index = bundle["targets"].index(target)
    per_run = bundle["counts"][index]
    cum = suffix_counts(per_run)
    return {
        "pos_cum": cum[:, positives, :].sum(axis=1).astype("float64"),
        "neg_cum": cum[:, negatives, :].sum(axis=1).astype("float64"),
        "pos_per_run": per_run[:, positives, :].sum(axis=(1, 2)).astype("float64"),
        "groups": bundle["groups"],
    }


def realign(arrays, groups, reference):
    """Reorder a run's per-run arrays onto another run's run ordering.

    Two runs are pairable only if they cover the same runs; the ordering they
    happen to have discovered them in is an artifact of fold order and must
    not leak into the comparison.
    """
    import numpy as np

    order = {label: position for position, label in enumerate(reference)}
    if set(order) != set(groups):
        raise CurveError("runs do not cover the same split_groups, so they are "
                         "not pairable")
    picked = np.asarray([groups.index(label) for label in reference], dtype="int64")
    return {key: value[picked] if key != "groups" else reference
            for key, value in arrays.items()}


def paired_difference(first, second, target, first_points, second_points,
                      iterations, seed, confidence):
    """AP and budgeted-recall differences with the shared run variation cancelled.

    Both models are scored on the **same** redrawn set of runs in every
    replicate, so the run-to-run variation they share drops out of the
    difference instead of being counted twice.  This is the only honest basis
    for saying one configuration ranks attacks better than another - two
    overlapping marginal intervals do not say it, which is the same argument
    `bootstrap_run_intervals.py` makes for macro F1 and the same reason this
    exists for AP.

    Each model keeps **its own** cross-fold thresholds at each budget: the
    question is which configuration is better when each is operated properly,
    not which is better at a threshold borrowed from the other.
    """
    import numpy as np

    a = target_arrays(first["bundle"], target)
    b = realign(target_arrays(second["bundle"], target),
                second["bundle"]["groups"], a["groups"])

    n_runs = len(a["groups"])
    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n_runs, size=(iterations, n_runs))
    ap = np.empty(iterations, dtype="float64")
    budgets = [point["budget"] for point in first_points]
    recall = {budget: np.empty(iterations, dtype="float64") for budget in budgets}

    order = {label: position for position, label in enumerate(a["groups"])}
    aligned_points = {}
    for budget in budgets:
        point_a = next(p for p in first_points if p["budget"] == budget)
        point_b = next(p for p in second_points if p["budget"] == budget)
        index_b = np.asarray([second["bundle"]["groups"].index(label)
                              for label in a["groups"]], dtype="int64")
        aligned_points[budget] = (point_a, point_b, index_b)

    for step in range(iterations):
        picked = draws[step]
        ap[step] = (
            average_precision(b["pos_cum"][picked].sum(axis=0),
                              b["neg_cum"][picked].sum(axis=0),
                              float(b["pos_per_run"][picked].sum()))
            - average_precision(a["pos_cum"][picked].sum(axis=0),
                                a["neg_cum"][picked].sum(axis=0),
                                float(a["pos_per_run"][picked].sum())))
        for budget, (point_a, point_b, index_b) in aligned_points.items():
            recall[budget][step] = (
                point_metrics(point_b, index_b[picked])["recall"]
                - point_metrics(point_a, picked)["recall"])

    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0

    def summarise(values, observed):
        lower = float(np.nanpercentile(values, lower_q))
        upper = float(np.nanpercentile(values, upper_q))
        return {"observed": observed, "lower": lower, "upper": upper,
                "separates": bool(lower > 0.0 or upper < 0.0)}

    observed_ap = (second["per_target"][target]["curve"]["ap"]
                   - first["per_target"][target]["curve"]["ap"])
    return {
        "target": target,
        "ap": summarise(ap, observed_ap),
        "recall": {
            budget: summarise(
                recall[budget],
                second["per_target"][target]["observed"][budget]["recall"]
                - first["per_target"][target]["observed"][budget]["recall"])
            for budget in budgets
        },
    }


def bootstrap_target(bundle, target, points, iterations, seed, confidence):
    """Run-level percentile intervals for AP and for every operating point.

    Each replicate redraws the runs with replacement and re-sums their
    per-run counts; the thresholds stay where the observed data put them
    (see the module docstring).  A target carried by fewer than
    `MIN_RUNS_FOR_INTERVAL` runs gets no interval, because resampling it
    would produce a zero-width one that reads as certainty.
    """
    import numpy as np

    positives, negatives = target_partition(bundle, target)
    index = bundle["targets"].index(target)
    per_run = bundle["counts"][index]
    cum = suffix_counts(per_run)
    pos_cum = cum[:, positives, :].sum(axis=1).astype("float64")
    neg_cum = cum[:, negatives, :].sum(axis=1).astype("float64")
    pos_per_run = per_run[:, positives, :].sum(axis=(1, 2))
    runs_with_positives = int((pos_per_run > 0).sum())

    n_runs = per_run.shape[0]
    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0
    estimable = runs_with_positives >= MIN_RUNS_FOR_INTERVAL
    result = {"runs_with_positives": runs_with_positives, "estimable": estimable}
    if not estimable:
        return result

    rng = np.random.RandomState(seed)
    draws = rng.randint(0, n_runs, size=(iterations, n_runs))
    ap = np.empty(iterations, dtype="float64")
    collected = {point["budget"]: {"recall": np.empty(iterations),
                                   "precision": np.empty(iterations),
                                   "alert_rate": np.empty(iterations)}
                 for point in points}
    for step in range(iterations):
        picked = draws[step]
        tp = pos_cum[picked].sum(axis=0)
        fp = neg_cum[picked].sum(axis=0)
        ap[step] = average_precision(tp, fp, float(pos_per_run[picked].sum()))
        for point in points:
            metrics = point_metrics(point, picked)
            for key in ("recall", "precision", "alert_rate"):
                collected[point["budget"]][key][step] = metrics[key]

    result["ap"] = {"lower": float(np.nanpercentile(ap, lower_q)),
                    "upper": float(np.nanpercentile(ap, upper_q))}
    result["budgets"] = {
        budget: {key: {"lower": float(np.nanpercentile(values[key], lower_q)),
                       "upper": float(np.nanpercentile(values[key], upper_q))}
                 for key in ("recall", "precision", "alert_rate")}
        for budget, values in collected.items()
    }
    return result


def effective_prior(report, prior):
    """Resolve `auto` against what the run was actually trained at.

    The goal is always "posteriors describing the deployment prior". For a
    `--balance none` run that is what they already are, so correcting would
    divide out a prior the model never had; for a balanced run it is a real
    correction. `auto` picks per run, which is what makes two differently
    balanced runs comparable in a single invocation - and a paired comparison
    between them meaningful.
    """
    if prior != "auto":
        return prior
    return "as-trained" if (report.get("balance") or "none") == "none" else "natural"


def analyse_run(directory, prior, selection, budgets, iterations, seed,
                confidence, edges):
    report = load_json(os.path.join(directory, "grouped_validation_report.json"))
    scores = os.path.join(directory, report.get("scores_file") or SCORES_FILENAME)
    if not os.path.exists(scores):
        raise CurveError(
            "%s has no %s - re-run run_grouped_validation.py with --save-scores"
            % (directory, SCORES_FILENAME))
    requested = prior
    prior = effective_prior(report, prior)
    weights = correction_weights(report, prior)
    bundle = accumulate_histograms(scores, report, weights, edges)

    thresholds = threshold_values(edges)
    per_target = {}
    for target in bundle["targets"]:
        points = [operating_point(bundle, target, budget, selection)
                  for budget in budgets]
        curve = pooled_curve(bundle, target)
        per_target[target] = {
            "curve": curve,
            "points": points,
            "observed": {point["budget"]: point_metrics(point) for point in points},
            "bootstrap": bootstrap_target(bundle, target, points, iterations,
                                          seed, confidence),
        }
    return {
        "label": os.path.basename(os.path.normpath(directory)),
        "directory": directory,
        "report": report,
        "bundle": bundle,
        "thresholds": thresholds,
        "prior": prior,
        "prior_requested": requested,
        "selection": selection,
        "per_target": per_target,
    }


def format_interval(value, interval, estimable=True, digits=4):
    import numpy as np

    if value is None or not np.isfinite(value):
        return "n/a"
    if not estimable or interval is None:
        return ("%%.%df (not estimable)" % digits) % value
    return ("%%.%df [%%.%df, %%.%df]" % (digits, digits, digits)) % (
        value, interval["lower"], interval["upper"])


def build_pairing_section(pairings, budgets, confidence):
    """What the marginal intervals above cannot settle."""
    if not pairings:
        return []
    lines = [
        "## Paired comparison",
        "",
        "Each replicate draws one set of runs and scores **both** configurations",
        "on it, so the run-to-run variation they share cancels instead of being",
        "counted twice. Two marginal intervals overlapping does not mean two",
        "configurations are indistinguishable; this is the table that settles it.",
        "Each configuration keeps its own cross-fold thresholds - the question is",
        "which is better when each is operated properly, not which wins at a",
        "threshold borrowed from the other.",
        "",
    ]
    for pairing in pairings:
        lines += [
            "### `%s` (A) vs `%s` (B)" % (pairing["first"], pairing["second"]),
            "",
            "| Target | B - A average precision | %.0f%% CI | separates? |"
            % (100 * confidence),
            "|---|---:|---|---|",
        ]
        for entry in pairing["targets"]:
            block = entry["ap"]
            lines.append("| `%s` | %+.4f | [%+.4f, %+.4f] | %s |" % (
                display_name(entry["target"]) if entry["target"] != ANY_ATTACK else ANY_ATTACK,
                block["observed"], block["lower"], block["upper"],
                "**yes**" if block["separates"] else "no - indistinguishable"))
        lines.append("")
        for budget in budgets:
            lines += [
                "B - A recall at %g alert(s) per 10,000 messages:" % (budget * 10000),
                "",
                "| Target | B - A recall | %.0f%% CI | separates? |" % (100 * confidence),
                "|---|---:|---|---|",
            ]
            for entry in pairing["targets"]:
                block = entry["recall"][budget]
                lines.append("| `%s` | %+.4f | [%+.4f, %+.4f] | %s |" % (
                    display_name(entry["target"]) if entry["target"] != ANY_ATTACK else ANY_ATTACK,
                    block["observed"], block["lower"], block["upper"],
                    "**yes**" if block["separates"] else "no"))
            lines.append("")
    return lines


def build_report(analyses, budgets, iterations, confidence, seed, edges,
                 generated=None, pairings=None):
    generated = generated or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Threshold curves over the grouped folds",
        "",
        f"- Generated: {generated}",
        f"- Threshold grid: {len(edges):,} points, uniform in the logit over "
        f"+/-{LOGIT_SPAN:g} (resolves to ~2e-9 from either end)",
        f"- Bootstrap: {iterations:,} replicates, percentile interval at "
        f"{confidence:.0%}, seed {seed}",
        "- **Resampling unit: `split_group` (one ERENO run)**, as in "
        "`bootstrap_run_intervals.py`.",
        "",
        "Every number in `grouped_validation_report.json` is `argmax(p)` - one",
        "point on the curves below, at whichever threshold the training prior",
        "put it. AP is the threshold-free summary; the budget tables are the",
        "operating points an alert burden actually allows.",
        "",
        "**Read AP against the prevalence column**, not against 1.0: a random",
        "detector scores AP = prevalence, so for `SAG.DB` at 0.219% of the pool",
        "the floor is 0.0022, not 0.",
        "",
    ]
    for analysis in analyses:
        report = analysis["report"]
        cap = report.get("max_train_rows_per_fold")
        lines += [
            "## `%s`" % analysis["label"],
            "",
            "- model: `%s` | balance: `%s` | train cap: %s | status: `%s`"
            % (report.get("model"), report.get("balance", "none"),
               f"{cap:,}" if cap else "—", report.get("status")),
            "- posterior prior: **%s** | threshold selection: **%s**"
            % ("corrected to the pool's natural prior" if analysis["prior"] == "natural"
               else "as trained (uncorrected)",
               "cross-fold (calibrated on the other folds)"
               if analysis["selection"] == "cross-fold"
               else "pooled (**optimistic - selected on the scored rows**)"),
            "- runs: %d | dataset SHA-256: `%s`"
            % (len(analysis["bundle"]["groups"]), report.get("dataset_sha256", "")[:16]),
            "",
            "### Threshold-free ranking",
            "",
            "| Target | positive rows | prevalence | runs carrying it | AP [%.0f%% CI] | AP / prevalence |"
            % (100 * confidence),
            "|---|---:|---:|---:|---|---:|",
        ]
        for target in analysis["bundle"]["targets"]:
            block = analysis["per_target"][target]
            curve, boot = block["curve"], block["bootstrap"]
            lift = (curve["ap"] / curve["prevalence"]) if curve["prevalence"] else float("nan")
            lines.append("| `%s` | %s | %.4f%% | %d | %s | %.1fx |" % (
                display_name(target) if target != ANY_ATTACK else ANY_ATTACK,
                f"{curve['positives']:,}", 100 * curve["prevalence"],
                boot["runs_with_positives"],
                format_interval(curve["ap"], boot.get("ap"), boot["estimable"]),
                lift))
        lines.append("")

        for budget in budgets:
            lines += [
                "### At %s alert%s per 10,000 messages (budget %g)"
                % (f"{budget * 10000:g}", "" if budget * 10000 == 1 else "s", budget),
                "",
                "| Target | threshold(s) | achieved alert rate | recall [%.0f%% CI] | precision [%.0f%% CI] | alerts | true |"
                % (100 * confidence, 100 * confidence),
                "|---|---|---:|---|---|---:|---:|",
            ]
            for target in analysis["bundle"]["targets"]:
                block = analysis["per_target"][target]
                observed = block["observed"][budget]
                boot = block["bootstrap"]
                intervals = boot.get("budgets", {}).get(budget, {})
                point = next(p for p in block["points"] if p["budget"] == budget)
                picks = sorted({int(v) for v in point["thresholds"].values()})
                # %.9g, not %.6g: the interesting thresholds on this problem
                # sit in the last thousandth of [0, 1], where fewer digits
                # print every distinct fold threshold as the same "1".
                shown = ", ".join("%.9g" % analysis["thresholds"][i] for i in picks[:3])
                if len(picks) > 3:
                    shown += ", ..."
                lines.append("| `%s` | %s | %.4f%% | %s | %s | %s | %s |" % (
                    display_name(target) if target != ANY_ATTACK else ANY_ATTACK,
                    shown, 100 * observed["alert_rate"],
                    format_interval(observed["recall"], intervals.get("recall"),
                                    boot["estimable"]),
                    format_interval(observed["precision"], intervals.get("precision"),
                                    boot["estimable"]),
                    f"{int(observed['tp'] + observed['fp']):,}",
                    f"{int(observed['tp']):,}"))
            lines.append("")

            lines += [
                "Where the false alarms come from, at this budget:",
                "",
                "| Target | " + " | ".join(
                    "`%s`" % display_name(name)
                    for name in analysis["bundle"]["classes"]) + " |",
                "|---" * (len(analysis["bundle"]["classes"]) + 1) + "|",
            ]
            for target in analysis["bundle"]["targets"]:
                block = analysis["per_target"][target]
                point = next(p for p in block["points"] if p["budget"] == budget)
                total_fp = float(point["fp"].sum())
                cells = []
                for name in analysis["bundle"]["classes"]:
                    source = point["fp_sources"].get(name)
                    if source is None:
                        cells.append("—")
                    elif total_fp <= 0:
                        cells.append("0")
                    else:
                        cells.append("%s (%.1f%%)" % (
                            f"{int(source.sum()):,}", 100 * source.sum() / total_fp))
                lines.append("| `%s` | %s |" % (
                    display_name(target) if target != ANY_ATTACK else ANY_ATTACK,
                    " | ".join(cells)))
            lines.append("")

    lines += build_pairing_section(pairings or [], budgets, confidence)
    lines += [
        "## How to read the budget tables",
        "",
        "- A threshold is picked on the **other** folds' rows and applied to the",
        "  fold being scored, so no recall here is inflated by having seen the",
        "  rows it is measured on. The per-fold thresholds are listed because",
        "  their spread *is* the calibration stability question.",
        "- `achieved alert rate` can exceed the budget. That means even the",
        "  strictest grid point alerts more often than the budget allows - the",
        "  model cannot be run that quietly, which is a result, not a rounding",
        "  problem.",
        "- `benign_degradation` in the false-alarm table is the card-C confound",
        "  on this axis: a detector whose alarms are mostly benign impairment is",
        "  keying on \"traffic looks degraded\", not on the attack.",
        "- The split between *ideal* `normal` and `normal` inside a benign run is",
        "  not made here - that rejoin lives in `benign_confusion_report.py`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def json_payload(analyses, budgets, iterations, confidence, seed, edges,
                 pairings=None):
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "grid": {"edges": len(edges), "logit_span": LOGIT_SPAN},
        "bootstrap": {"iterations": iterations, "confidence": confidence, "seed": seed},
        "budgets": list(budgets),
        "runs": [],
    }
    for analysis in analyses:
        entry = {
            "label": analysis["label"],
            "model": analysis["report"].get("model"),
            "balance": analysis["report"].get("balance"),
            "dataset_sha256": analysis["report"].get("dataset_sha256"),
            "prior": analysis["prior"],
            "prior_requested": analysis["prior_requested"],
            "threshold_selection": analysis["selection"],
            "n_runs": len(analysis["bundle"]["groups"]),
            "targets": {},
        }
        for target in analysis["bundle"]["targets"]:
            block = analysis["per_target"][target]
            boot = block["bootstrap"]
            entry["targets"][target] = {
                "positives": block["curve"]["positives"],
                "prevalence": block["curve"]["prevalence"],
                "average_precision": block["curve"]["ap"],
                "average_precision_ci": boot.get("ap"),
                "runs_with_positives": boot["runs_with_positives"],
                "estimable": boot["estimable"],
                "operating_points": [
                    {
                        "budget": point["budget"],
                        "thresholds": {
                            fold: float(analysis["thresholds"][index])
                            for fold, index in point["thresholds"].items()
                        },
                        "observed": {
                            key: value for key, value in
                            block["observed"][point["budget"]].items()
                        },
                        "ci": boot.get("budgets", {}).get(point["budget"]),
                        "false_alarm_sources": {
                            name: int(source.sum())
                            for name, source in point["fp_sources"].items()
                        },
                    }
                    for point in block["points"]
                ],
            }
        payload["runs"].append(entry)
    payload["pairings"] = pairings or []
    return payload


def write_curve_csv(path, analyses, thresholds):
    """Every grid point, for plotting outside this script."""
    import csv as csv_module

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv_module.writer(fh)
        writer.writerow(["run", "target", "threshold", "tp", "fp",
                         "recall", "precision", "alert_rate"])
        for analysis in analyses:
            for target in analysis["bundle"]["targets"]:
                curve = analysis["per_target"][target]["curve"]
                tp, fp = curve["tp"], curve["fp"]
                positives, rows = curve["positives"], curve["rows"]
                for index in range(len(thresholds)):
                    alerted = tp[index] + fp[index]
                    writer.writerow([
                        analysis["label"], target, "%.9g" % thresholds[index],
                        int(tp[index]), int(fp[index]),
                        "%.6f" % (tp[index] / positives) if positives else "",
                        "%.6f" % (tp[index] / alerted) if alerted else "",
                        "%.9f" % (alerted / rows) if rows else "",
                    ])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="A run directory holding grouped_validation_report.json "
                             "and grouped_scores.parquet. Repeatable.")
    parser.add_argument("--prior", choices=["auto", "as-trained", "natural"],
                        default="auto",
                        help="'natural' divides out the training prior so the posteriors "
                             "describe the pool's own class balance; 'as-trained' leaves "
                             "them alone. 'auto' (default) picks per run from its recorded "
                             "balance, which is what lets a balanced and an unbalanced run "
                             "be compared in one invocation.")
    parser.add_argument("--threshold-selection", choices=["cross-fold", "pooled"],
                        default="cross-fold",
                        help="'cross-fold' calibrates each fold's threshold on the other "
                             "folds. 'pooled' selects on the scored rows themselves and "
                             "is optimistic - for measuring that gap, not for reporting.")
    parser.add_argument("--budget", action="append", type=float, default=None,
                        help="Alert budget as a fraction of all evaluated rows "
                             "(default: %s). Repeatable."
                             % ", ".join(str(b) for b in DEFAULT_BUDGETS))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--dataset", default=None,
                        help="Optional: re-verify the run's dataset SHA-256 before "
                             "reporting from it.")
    parser.add_argument("--out", default=os.path.join(HERE, "pr_curves.md"))
    parser.add_argument("--json-out", default=None,
                        help="Defaults to --out with a .json suffix.")
    parser.add_argument("--curve-csv", default=None,
                        help="Optional: every grid point, for plotting.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    budgets = sorted(set(args.budget or list(DEFAULT_BUDGETS)))
    try:
        if not 0.0 < args.confidence < 1.0:
            raise CurveError("--confidence must be strictly between 0 and 1")
        if any(not 0.0 < b <= 1.0 for b in budgets):
            raise CurveError("every --budget must be in (0, 1]")
        edges = threshold_edges()
        analyses = []
        for directory in args.run:
            analysis = analyse_run(directory, args.prior, args.threshold_selection,
                                   budgets, args.iterations, args.seed,
                                   args.confidence, edges)
            if args.dataset:
                digest = sha256_file(args.dataset)
                if digest != analysis["report"].get("dataset_sha256"):
                    raise CurveError(
                        "%s was produced on a different dataset (%s != %s)"
                        % (directory, digest[:16],
                           str(analysis["report"].get("dataset_sha256"))[:16]))
            analyses.append(analysis)

        pairings = []
        for analysis in analyses[1:]:
            shared = [t for t in analyses[0]["bundle"]["targets"]
                      if t in analysis["bundle"]["targets"]]
            pairings.append({
                "first": analyses[0]["label"], "second": analysis["label"],
                "targets": [
                    paired_difference(
                        analyses[0], analysis, target,
                        analyses[0]["per_target"][target]["points"],
                        analysis["per_target"][target]["points"],
                        args.iterations, args.seed, args.confidence)
                    for target in shared
                ],
            })

        text = build_report(analyses, budgets, args.iterations, args.confidence,
                            args.seed, edges, pairings=pairings)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        json_out = args.json_out or (os.path.splitext(args.out)[0] + ".json")
        with open(json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(json_payload(analyses, budgets, args.iterations,
                                   args.confidence, args.seed, edges,
                                   pairings=pairings), fh, indent=2)
            fh.write("\n")
        if args.curve_csv:
            write_curve_csv(args.curve_csv, analyses, threshold_values(edges))
    except (OSError, json.JSONDecodeError, CurveError, ValueError, KeyError) as exc:
        print("PR CURVE REPORT FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print("Wrote %s and %s (%d run(s), %d budget(s))."
          % (args.out, json_out, len(analyses), len(budgets)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
