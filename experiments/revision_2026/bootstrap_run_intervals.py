"""Confidence intervals that respect the experimental unit (the run).

Checklist D.5/E.4.  The per-fold spread this pipeline reports is *not* five
independent estimates: `split_group` (one ERENO run) is the independent unit,
and a 5-fold grouped split just partitions the runs.  When a class has few
runs the difference is not cosmetic - `SAG.DB` and `FRG` have 15 runs each, so
a fold's test partition can hold a *single* run and its per-fold recall is a
measurement of that one run's ~1,000 correlated messages (see
`validation_protocol.md`, "How many attack rows are actually being counted").

This script therefore resamples **runs**, not rows and not folds.  Each
bootstrap replicate draws `n` runs with replacement from the `n` runs the
evaluation covered, sums their confusion matrices, and recomputes every metric
from that total.  Percentile intervals over those replicates are the honest
statement of how much the reported numbers depend on which runs happened to be
generated - which is exactly the quantity a reviewer asks about when a class
rests on 15 runs.

Deliberately *not* a bootstrap over rows: rows inside a run are correlated by
construction (one publisher stream, retransmissions of the same event), so
resampling them would produce intervals that are far too narrow - the same
mistake at metric level that message-level splitting makes at protocol level.

**The method has a floor, and it is reported rather than hidden.** A class
carried by a single run has *no* resamplable variation: every replicate that
contains that run reproduces its metric exactly, so the percentile interval
collapses to zero width. That would read as perfect certainty while meaning
the exact opposite, so such classes are reported as ``not estimable`` instead
of being given an interval. Two runs is the bare minimum for any spread at
all, and intervals from a handful of runs are coarse by construction - they
move in steps of roughly one run's worth of the metric.

Usage mirrors `check_prediction_integrity.py`:

    python experiments/revision_2026/bootstrap_run_intervals.py \\
      --run results/d3-xgboost-downsample \\
      --out experiments/revision_2026/run_bootstrap.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED_COLUMNS = ["split_group", "y_true", "y_pred"]

# Below this many runs carrying a class there is nothing for a run-level
# bootstrap to resample, so no interval is reported for it (see module
# docstring). `THIN_RUNS` is not a correctness threshold but a reading aid:
# intervals built on fewer runs than this are coarse and should be read as
# such.
MIN_RUNS_FOR_INTERVAL = 2
THIN_RUNS = 20


class BootstrapError(ValueError):
    pass


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def per_run_confusion(path, classes, chunk_rows=2_000_000):
    """Stream the predictions into one confusion matrix per run.

    The result is tiny - ``n_runs x n_classes x n_classes`` int64, a few
    hundred kilobytes even for 265 runs - which is what makes resampling
    cheap afterwards: a replicate is a sum over a slice of this array rather
    than another pass over 20M rows.
    """
    import numpy as np
    import pandas as pd

    class_to_code = {name: index for index, name in enumerate(classes)}
    n = len(classes)
    groups = {}
    counts = []
    unknown = set()

    for chunk in pd.read_csv(path, usecols=REQUIRED_COLUMNS,
                             chunksize=chunk_rows, encoding="utf-8"):
        missing = set(REQUIRED_COLUMNS) - set(chunk.columns)
        if missing:
            raise BootstrapError("predictions CSV is missing columns: %s" % sorted(missing))
        true_codes = chunk["y_true"].map(class_to_code)
        pred_codes = chunk["y_pred"].map(class_to_code)
        if true_codes.isna().any() or pred_codes.isna().any():
            unknown |= set(chunk.loc[true_codes.isna(), "y_true"].unique())
            unknown |= set(chunk.loc[pred_codes.isna(), "y_pred"].unique())
            continue
        for group, block in chunk.groupby("split_group", sort=False, observed=True):
            index = groups.get(group)
            if index is None:
                index = groups[group] = len(counts)
                counts.append(np.zeros((n, n), dtype=np.int64))
            t = true_codes.loc[block.index].to_numpy(dtype=np.int64)
            p = pred_codes.loc[block.index].to_numpy(dtype=np.int64)
            counts[index] += np.bincount(t * n + p, minlength=n * n).reshape(n, n)

    if unknown:
        raise BootstrapError(
            "predictions contain labels absent from the report's class list: %s"
            % sorted(str(c) for c in unknown))
    if not counts:
        raise BootstrapError("predictions CSV held no rows")
    return list(groups), np.stack(counts)


def metrics_from_matrix(matrix):
    """Per-class precision/recall/F1 plus macro F1, from one confusion matrix.

    Rows are truth, columns are prediction.  Two different kinds of
    "undefined" have to be kept apart here, and conflating them silently
    inflates macro F1:

    * **The class is absent from this matrix** (``support == 0``) - a
      bootstrap replicate that drew no run carrying it. There is genuinely no
      information, so every metric is NaN and the class is left out of the
      macro average. Scoring it 0 would punish the model for a class the
      replicate never tested.
    * **The class is present but the model never predicted it**
      (``support > 0``, ``predicted == 0``). That is not missing information,
      it is a total failure on that class: recall is 0 by measurement, and
      precision and F1 are 0 by the ``zero_division=0`` convention the rest of
      this pipeline uses (`sklearn`'s ``classification_report``). The class
      *must* stay in the macro average - dropping it is how a model that
      detects nothing ends up with a flattering macro F1.

    Keeping the second case in is what makes this script's macro F1 agree with
    the figure `run_grouped_validation.py` recorded.
    """
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        tp = np.diagonal(matrix, axis1=-2, axis2=-1).astype(float)
        support = matrix.sum(axis=-1).astype(float)
        predicted = matrix.sum(axis=-2).astype(float)
        present = support > 0
        recall = np.where(present, tp / np.where(present, support, 1), np.nan)
        # zero_division=0 for a present-but-never-predicted class; NaN only
        # when the class is absent altogether.
        precision = np.where(
            predicted > 0, tp / np.where(predicted > 0, predicted, 1),
            np.where(present, 0.0, np.nan))
        denominator = precision + recall
        f1 = np.where(denominator > 0,
                      2 * precision * recall / np.where(denominator > 0, denominator, 1), 0.0)
        f1 = np.where(present, f1, np.nan)
        total = matrix.sum(axis=(-2, -1)).astype(float)
        accuracy = np.where(total > 0, tp.sum(axis=-1) / np.where(total > 0, total, 1), np.nan)
    with np.errstate(invalid="ignore"):
        macro_f1 = np.nanmean(f1, axis=-1)
    return {"precision": precision, "recall": recall, "f1": f1,
            "macro_f1": macro_f1, "accuracy": accuracy}


def bootstrap(counts, iterations, seed, confidence):
    """Percentile intervals from resampling runs with replacement."""
    import numpy as np

    rng = np.random.RandomState(seed)
    n_runs = counts.shape[0]
    draws = rng.randint(0, n_runs, size=(iterations, n_runs))
    # counts[draws] would be iterations x n_runs x C x C - for 2000 x 265 x 6 x 6
    # that is ~2.3 GB. Accumulating a replicate at a time keeps it at one C x C.
    replicate_matrices = np.empty((iterations,) + counts.shape[1:], dtype=np.int64)
    for index in range(iterations):
        replicate_matrices[index] = counts[draws[index]].sum(axis=0)
    stats = metrics_from_matrix(replicate_matrices)
    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0
    out = {}
    for key, values in stats.items():
        finite = np.isfinite(values)
        out[key] = {
            "lower": np.nanpercentile(np.where(finite, values, np.nan), lower_q, axis=0),
            "upper": np.nanpercentile(np.where(finite, values, np.nan), upper_q, axis=0),
            "undefined_replicates": (~finite).sum(axis=0),
        }
    return out


def paired_difference(first, second, metric, iterations, seed, confidence):
    """Bootstrap the *difference* between two runs over the same resampled runs.

    Comparing two marginal intervals and asking whether they overlap is the
    classic way to miss a real difference: both models are evaluated on the
    *same* runs, so the run-to-run variation they share cancels when the
    difference is taken draw by draw. This resamples the run indices once per
    replicate and scores both models on that same draw - the paired analogue
    of the agreement tables `check_prediction_integrity.py` builds.

    Returns the observed difference (second minus first), its percentile
    interval, and whether that interval excludes zero, which is the only
    honest basis for saying one model beat the other.
    """
    import numpy as np

    order = {label: index for index, label in enumerate(first["groups"])}
    if set(order) != set(second["groups"]):
        raise BootstrapError(
            "runs %r and %r do not cover the same runs, so they are not pairable"
            % (first["label"], second["label"]))
    # Align the second run's per-run matrices onto the first run's ordering.
    realigned = np.empty_like(second["counts"])
    for index, label in enumerate(second["groups"]):
        realigned[order[label]] = second["counts"][index]

    rng = np.random.RandomState(seed)
    n_runs = first["counts"].shape[0]
    draws = rng.randint(0, n_runs, size=(iterations, n_runs))
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picked = draws[index]
        a = metrics_from_matrix(first["counts"][picked].sum(axis=0))[metric]
        b = metrics_from_matrix(realigned[picked].sum(axis=0))[metric]
        differences[index] = float(b) - float(a)
    lower_q = (1.0 - confidence) / 2.0 * 100.0
    upper_q = (1.0 + confidence) / 2.0 * 100.0
    lower = float(np.nanpercentile(differences, lower_q))
    upper = float(np.nanpercentile(differences, upper_q))
    observed = (float(metrics_from_matrix(realigned.sum(axis=0))[metric])
                - float(metrics_from_matrix(first["counts"].sum(axis=0))[metric]))
    return {
        "first": first["label"], "second": second["label"], "metric": metric,
        "observed_difference": observed, "lower": lower, "upper": upper,
        "separates": bool(lower > 0.0 or upper < 0.0),
    }


def audit_run(directory, iterations, seed, confidence):
    import numpy as np

    report = load_json(os.path.join(directory, "grouped_validation_report.json"))
    classes = report["classes"]
    predictions = os.path.join(directory, "grouped_predictions.csv")
    if not os.path.exists(predictions):
        raise BootstrapError("no grouped_predictions.csv in %s" % directory)
    groups, counts = per_run_confusion(predictions, classes)
    observed = metrics_from_matrix(counts.sum(axis=0))
    intervals = bootstrap(counts, iterations, seed, confidence)
    runs_with_class = (counts.sum(axis=2) > 0).sum(axis=0)
    estimable = runs_with_class >= MIN_RUNS_FOR_INTERVAL
    return {
        "estimable": estimable,
        "label": os.path.basename(os.path.normpath(directory)),
        "directory": directory,
        "groups": groups,
        "counts": counts,
        "model": report.get("model"),
        "balance": report.get("balance", "none"),
        "max_train_rows_per_fold": report.get("max_train_rows_per_fold"),
        "classes": classes,
        "n_runs": len(groups),
        "runs_with_class": runs_with_class,
        "support": counts.sum(axis=0).sum(axis=1),
        "observed": observed,
        "intervals": intervals,
    }


def format_interval(value, lower, upper, estimable=True):
    import numpy as np
    if not np.isfinite(value):
        return "n/a"
    if not estimable:
        # A zero-width interval here would mean "one run, nothing to
        # resample", which reads as certainty and means the opposite.
        return "%.4f (not estimable)" % value
    return "%.4f [%.4f, %.4f]" % (value, lower, upper)


def build_pairing_section(pairings, confidence):
    """The comparison the marginal intervals above cannot make."""
    if not pairings:
        return []
    lines = [
        "## Paired comparison (macro F1)",
        "",
        "Each replicate draws one set of runs and scores **both** models on it, so",
        "the run-to-run variation the two share cancels instead of being counted",
        "twice. Two marginal intervals overlapping does not mean two models are",
        "indistinguishable, and this table is what actually settles it.",
        "",
        "| A | B | B - A | %.0f%% CI | separates? |" % (100 * confidence),
        "|---|---|---:|---|---|",
    ]
    for pairing in pairings:
        lines.append("| `%s` | `%s` | %+.4f | [%+.4f, %+.4f] | %s |" % (
            pairing["first"], pairing["second"], pairing["observed_difference"],
            pairing["lower"], pairing["upper"],
            "**yes**" if pairing["separates"] else "no - indistinguishable"))
    lines.append("")
    return lines


def build_report(runs, iterations, confidence, seed, generated=None, pairings=None):
    generated = generated or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Run-level bootstrap intervals",
        "",
        f"- Generated: {generated}",
        f"- Replicates: {iterations:,}, percentile interval at {confidence:.0%}, seed {seed}",
        "- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**",
        "  Rows inside a run are correlated by construction, so a row-level bootstrap",
        "  would report intervals far narrower than the evidence supports.",
        "",
        "An interval here answers: *if the run matrix had drawn a different set of",
        "runs from the same generator, how much would this number move?* It does not",
        "capture uncertainty from the generator's design itself (attack prevalence,",
        "loss rates and burst sizes are fixed by the matrix, not sampled).",
        "",
    ]
    for run in runs:
        cap = run["max_train_rows_per_fold"]
        lines += [
            "## `%s`" % run["label"],
            "",
            "- model: `%s` | balance: `%s` | train cap: %s | runs covered: **%d**"
            % (run["model"], run["balance"], f"{cap:,}" if cap else "—", run["n_runs"]),
            "",
            "| Class | runs carrying it | rows | recall [%.0f%% CI] | precision [%.0f%% CI] | F1 [%.0f%% CI] |"
            % (100 * 0.95, 100 * 0.95, 100 * 0.95),
            "|---|---:|---:|---|---|---|",
        ]
        for index, name in enumerate(run["classes"]):
            estimable = bool(run["estimable"][index])
            lines.append("| `%s` | %d | %s | %s | %s | %s |" % (
                name,
                int(run["runs_with_class"][index]),
                f"{int(run['support'][index]):,}",
                format_interval(run["observed"]["recall"][index],
                                run["intervals"]["recall"]["lower"][index],
                                run["intervals"]["recall"]["upper"][index], estimable),
                format_interval(run["observed"]["precision"][index],
                                run["intervals"]["precision"]["lower"][index],
                                run["intervals"]["precision"]["upper"][index], estimable),
                format_interval(run["observed"]["f1"][index],
                                run["intervals"]["f1"]["lower"][index],
                                run["intervals"]["f1"]["upper"][index], estimable),
            ))
        lines += [
            "",
            "- macro F1: %s" % format_interval(
                float(run["observed"]["macro_f1"]),
                float(run["intervals"]["macro_f1"]["lower"]),
                float(run["intervals"]["macro_f1"]["upper"])),
            "- accuracy (micro): %s" % format_interval(
                float(run["observed"]["accuracy"]),
                float(run["intervals"]["accuracy"]["lower"]),
                float(run["intervals"]["accuracy"]["upper"])),
            "",
        ]
        not_estimable = [run["classes"][i] for i in range(len(run["classes"]))
                         if not run["estimable"][i]]
        if not_estimable:
            lines += [
                "> **Not estimable:** %s is carried by fewer than %d runs, so a run-level"
                % (", ".join("`%s`" % name for name in not_estimable), MIN_RUNS_FOR_INTERVAL),
                "> bootstrap has nothing to resample for it. The point estimate is shown",
                "> without an interval; a zero-width interval would read as certainty and",
                "> mean the opposite.",
                "",
            ]
        thin = [run["classes"][i] for i in range(len(run["classes"]))
                if run["estimable"][i] and run["runs_with_class"][i] < THIN_RUNS]
        if thin:
            lines += [
                "> **Thin classes:** %s carry fewer than %d independent runs, so their"
                % (", ".join("`%s`" % name for name in thin), THIN_RUNS),
                "> intervals are wide, and coarse - they move in steps of roughly one",
                "> run's worth of the metric. That is thin evidence, not an unstable",
                "> model. Narrowing them needs more runs of those variants, not more rows",
                "> per run.",
                "",
            ]
    lines += build_pairing_section(pairings or [], confidence)
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True,
                        help="A finished run directory; repeatable.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 < args.confidence < 1.0:
        print("BOOTSTRAP FAILED\n--confidence must be in (0, 1)", file=sys.stderr)
        return 1
    if args.iterations < 1:
        print("BOOTSTRAP FAILED\n--iterations must be positive", file=sys.stderr)
        return 1
    try:
        runs = [audit_run(directory, args.iterations, args.seed, args.confidence)
                for directory in args.run]
        # Paired against the first run, matching check_prediction_integrity.py's
        # convention so the two reports line up run for run.
        pairings = [
            paired_difference(runs[0], other, "macro_f1",
                              args.iterations, args.seed, args.confidence)
            for other in runs[1:]
        ]
        out_path = args.out or os.path.join(HERE, "run_bootstrap.md")
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(build_report(runs, args.iterations, args.confidence, args.seed,
                                  pairings=pairings))
        print("Run-level bootstrap written: %s" % out_path)
        for run in runs:
            thin = sum(1 for i in range(len(run["classes"]))
                       if run["runs_with_class"][i] < THIN_RUNS)
            blind = sum(1 for i in range(len(run["classes"])) if not run["estimable"][i])
            print("  %s: %d runs, %d class(es) on fewer than %d runs, %d not estimable"
                  % (run["label"], run["n_runs"], thin, THIN_RUNS, blind))
        if args.json_out:
            payload = {
                "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "iterations": args.iterations,
                "confidence": args.confidence,
                "seed": args.seed,
                "resampling_unit": "split_group",
                "runs": [{
                    "label": run["label"],
                    "directory": run["directory"],
                    "model": run["model"],
                    "balance": run["balance"],
                    "max_train_rows_per_fold": run["max_train_rows_per_fold"],
                    "n_runs": run["n_runs"],
                    "classes": run["classes"],
                    "runs_with_class": [int(v) for v in run["runs_with_class"]],
                    "interval_estimable": [bool(v) for v in run["estimable"]],
                    "min_runs_for_interval": MIN_RUNS_FOR_INTERVAL,
                    "support": [int(v) for v in run["support"]],
                    "observed": {
                        key: ([float(v) for v in value] if getattr(value, "ndim", 0) else float(value))
                        for key, value in run["observed"].items()
                    },
                    "intervals": {
                        key: {
                            bound: ([float(v) for v in value] if getattr(value, "ndim", 0) else float(value))
                            for bound, value in bounds.items()
                        } for key, bounds in run["intervals"].items()
                    },
                } for run in runs],
                "pairings": pairings,
            }
            with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, indent=2, default=float)
                fh.write("\n")
            print("JSON written: %s" % args.json_out)
    except (OSError, json.JSONDecodeError, BootstrapError, ValueError) as exc:
        print("BOOTSTRAP FAILED\n%s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
