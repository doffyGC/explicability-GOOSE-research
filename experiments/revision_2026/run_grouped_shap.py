"""Card F.3/F.4: SHAP on held-out grouped data, with background from train.

Preregistered in `explainability_card.md` before a single value was computed;
that document owns the protocol, the expectations and the falsification
conditions, and this script owns making them true.

The defect it closes
--------------------
`explainability/shap_analysis.py` in the frozen baseline calls
`shap.Explainer(model)` on the **last CV fold's validation set**. Under
message-level `StratifiedKFold` that set holds messages from the runs the
model was fitted on, so every importance in the submitted paper was measured
partly on training data. It is the last of the four defects that motivated
this revision still standing.

Here, SHAP is computed on rows from each fold's **test** partition of the
persisted, hash-bound `splits_grouped.json` - the same folds every card-D
number comes from - and the background distribution comes from the **train**
partition of the *same* fold. That pairing is what F.3 asks for: the
attribution is against what this model was fitted on, not against the rows
being explained.

The hazard, and the check that removes it
-----------------------------------------
No run in `results/` persists a fitted estimator, so this card has to refit.
A SHAP value computed on a *different* model than the published one explains
nothing about the paper, and "same seed, same data, therefore same model" is
an assumption that costs nothing to verify and everything to get wrong.

So the refit is reconstructed exactly as `run_grouped_validation.run_folds`
builds it - same persisted fold, same `seed + fold_index`, same feature set,
same library defaults, same `n_jobs` (XGBoost's histogram reduction order
depends on the thread count) - and **its predictions on the test partition
must reproduce the reference run's `grouped_predictions.csv` row for row**.
A single mismatched row is fatal. `--skip-verification` exists for a
reference run whose predictions are unavailable, and marks the output as
unverified everywhere it appears.

What is computed, and what is deliberately not
----------------------------------------------
Per-class SHAP values, kept per class and never summed into one
"importance": a feature separating `SAG.DB` from `SAG.PB` and a feature
separating `normal` from everything are not the same finding
(`ablations_baselines.md` §5). The reported quantity is mean |SHAP| per
feature, per class, **per fold** - the spread across folds is F.4's
deliverable and is shown rather than averaged away.

Two aggregations, because at 2% attack prevalence one of them answers a
question nobody asked:

- **global** - mean |SHAP| over every explained row. This is the standard
  global importance, and on this pool it is measured on ~95.6% `normal`
  traffic, because the explained sample is a proportional replica of the
  held-out distribution. "What moves the `SAG.DB` output across held-out
  traffic" is a real quantity, and it is mostly a statement about normal
  messages.
- **conditional** - mean |SHAP| over the rows whose *true* class is a given
  one. "When the model is looking at an actual `SAG.DB` message, what drives
  its `SAG.DB` score" is the question an explainability claim in this paper
  is actually making, and it is not recoverable from the global average.

Both are written. The conditional one rests on far fewer rows - ~90-120 per
attack class per fold, from ~9 runs - so its support is reported in the same
table as its values, and the fold-to-fold spread is what says whether a
ranking survives that thinness.

`feature_perturbation="interventional"` is not a default worth inheriting
silently: `tree_path_dependent` ignores the background entirely and answers a
different question, so a run that fell back to it would satisfy F.3's letter
and not its content. The explainer is constructed with the background and the
choice is recorded in the report.

**This script draws no conclusion about causation.** SHAP attributes a
model's output to its inputs; whether a feature *causes* anything in the
substation is not tested here (card F.5, `explainability_card.md` §7).

Usage
-----
    python experiments/revision_2026/run_grouped_shap.py \\
      --dataset data/runs/gray-GOOSE-runs-prepared.parquet \\
      --preparation-report experiments/revision_2026/preparation_audit.json \\
      --splits experiments/revision_2026/splits_grouped.json \\
      --reference-run results/v2-xgboost-none \\
      --out-dir results/f3-xgboost-shap
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

from run_grouped_validation import (
    GroupedRunError,
    classifier,
    load_inputs,
    load_json,
    predict_in_chunks,
    provenance_block,
    subsample_train,
)

HERE = os.path.dirname(os.path.abspath(__file__))

REPORT_NAME = "grouped_validation_report.json"
PREDICTIONS_NAME = "grouped_predictions.csv"
IMPORTANCES_NAME = "shap_importances.json"

CHUNK_ROWS = 2_000_000


class ShapRunError(ValueError):
    pass


# --------------------------------------------------------------------------
# The reference run this card explains
# --------------------------------------------------------------------------

def load_reference(directory, digest, splits_path):
    """The published run whose model this card refits, and its predictions.

    Bound by dataset hash and split file, because explaining a model fitted
    on another pool - or on other folds - with this pool's rows would be a
    silent category error rather than a visible failure.
    """
    report_path = os.path.join(directory, REPORT_NAME)
    if not os.path.exists(report_path):
        raise ShapRunError("%s has no %s" % (directory, REPORT_NAME))
    report = load_json(report_path)

    if report.get("dataset_sha256") != digest:
        raise ShapRunError(
            "reference run was produced on dataset %s but this card is running "
            "on %s" % (str(report.get("dataset_sha256"))[:12], str(digest)[:12]))
    if os.path.basename(str(report.get("splits"))) != os.path.basename(splits_path):
        raise ShapRunError(
            "reference run consumed split file %r, not %r - the folds must be "
            "the same folds" % (os.path.basename(str(report.get("splits"))),
                                os.path.basename(splits_path)))
    if report.get("status") != "full_grouped_run":
        raise ShapRunError(
            "reference run is %r; a technical smoke is a wiring check, not the "
            "published model" % report.get("status"))
    return report


def reference_predictions(directory, classes):
    """`split_id -> (row_index, predicted code)`, read in chunks.

    The predictions file is ~722 MB. Only the row index and the predicted
    label are needed, and the label is mapped to its class code on the way in
    so what is held is two compact arrays per fold rather than 11M strings.
    """
    import numpy as np
    import pandas as pd

    path = os.path.join(directory, PREDICTIONS_NAME)
    if not os.path.exists(path):
        raise ShapRunError("%s has no %s" % (directory, PREDICTIONS_NAME))

    code_of = {name: code for code, name in enumerate(classes)}
    collected = {}
    # Closed explicitly: the loop below can raise part way through a 722 MB
    # file, and an abandoned reader leaves the handle open until the
    # interpreter gets around to it.
    with pd.read_csv(path, usecols=["split_id", "row_index", "y_pred"],
                     encoding="utf-8", chunksize=CHUNK_ROWS) as reader:
        for chunk in reader:
            unknown = set(chunk["y_pred"].unique()) - set(code_of)
            if unknown:
                raise ShapRunError(
                    "reference predictions carry labels the dataset does not "
                    "declare: %s" % sorted(unknown))
            codes = chunk["y_pred"].map(code_of).to_numpy(dtype="int16")
            rows = chunk["row_index"].to_numpy(dtype="int64")
            for split_id, group in chunk.groupby("split_id", sort=False).indices.items():
                bucket = collected.setdefault(split_id, ([], []))
                bucket[0].append(rows[group])
                bucket[1].append(codes[group])
    return {split_id: (np.concatenate(rows), np.concatenate(codes))
            for split_id, (rows, codes) in collected.items()}


def verify_fold(split_id, row_index, test_positions, predicted, reference):
    """The refit model must be the published model, row for row.

    Compared on `row_index` rather than on position, because the two are the
    same ordering only as long as nothing upstream changes - and this check
    exists precisely to catch the cases where something did.
    """
    import numpy as np

    if split_id not in reference:
        raise ShapRunError(
            "%s is missing from the reference predictions" % split_id)
    ref_rows, ref_codes = reference[split_id]
    rows = row_index[test_positions]

    if len(rows) != len(ref_rows):
        raise ShapRunError(
            "%s: refit evaluated %d rows, the reference run %d"
            % (split_id, len(rows), len(ref_rows)))

    order = np.argsort(ref_rows, kind="stable")
    mine = np.argsort(rows, kind="stable")
    if not np.array_equal(ref_rows[order], rows[mine]):
        raise ShapRunError(
            "%s: refit and reference cover different rows" % split_id)

    mismatches = int(np.count_nonzero(ref_codes[order] != predicted[mine]))
    if mismatches:
        raise ShapRunError(
            "%s: the refit model disagrees with the published run on %d of %d "
            "rows. The SHAP values would explain a different model than the "
            "paper reports, so this card stops here."
            % (split_id, mismatches, len(rows)))
    return len(rows)


# --------------------------------------------------------------------------
# Folds, rebuilt exactly as the runner builds them
# --------------------------------------------------------------------------

def fold_positions(arrays, split, seed, fold_index, max_train_rows):
    """Train and test positions for one persisted fold.

    Mirrors `run_grouped_validation.run_folds` step for step - the same mask,
    the same overlap and empty-partition refusals, the same
    `subsample_train(..., seed + fold_index)` - because a refit that built its
    train partition even slightly differently would not be the published
    model, and the verification downstream would then be failing for a reason
    that has nothing to do with the question.
    """
    import numpy as np

    group_codes = arrays["group_codes"]
    group_labels = arrays["group_labels"]
    y = arrays["y"]
    classes = np.asarray(arrays["classes"], dtype=object)
    label_to_code = {label: code for code, label in enumerate(group_labels)}

    def mask_for(partition):
        codes = [label_to_code[g] for g in split[partition] if g in label_to_code]
        return np.isin(group_codes, np.asarray(codes, dtype=group_codes.dtype))

    train_mask = mask_for("train_groups")
    test_mask = mask_for("test_groups")
    if (train_mask & test_mask).any():
        raise GroupedRunError("%s has row-level train/test overlap" % split["split_id"])
    if not train_mask.any() or not test_mask.any():
        raise GroupedRunError("%s has an empty partition" % split["split_id"])

    train_positions = np.flatnonzero(train_mask)
    strata = (group_codes[train_positions].astype("int32") * len(classes)
              + y[train_positions])
    keep = subsample_train(strata, max_train_rows, seed + fold_index)
    if keep is not None:
        train_positions = train_positions[keep]
    test_positions = np.flatnonzero(test_mask)
    del train_mask, test_mask, strata, keep
    gc.collect()
    return train_positions, test_positions


def stratified_sample(positions, group_codes, y, n_classes, cap, seed):
    """A (`split_group`, `class`)-stratified sample of the rows to explain.

    The same `subsample_train` the runner uses for its documented train cap,
    pointed at the rows being explained: every test run of the fold keeps its
    share, no rare class can be emptied, and what shrinks is rows per run -
    which is the property `explainability_card.md` §3 preregistered.

    Note the floor this carries, measured rather than assumed: the sampler
    cannot return fewer rows than it has non-empty strata, so on a fold with
    ~53 test runs and six classes it will not go below ~300 rows whatever cap
    it is given. That is the right behaviour here and the wrong behaviour for
    a background distribution - see `background_sample`.
    """
    if not cap or len(positions) <= cap:
        return positions
    strata = (group_codes[positions].astype("int32") * n_classes + y[positions])
    keep = subsample_train(strata, cap, seed)
    return positions if keep is None else positions[keep]


def background_sample(positions, y, cap, seed):
    """A class-stratified sample of train rows, of the size actually asked for.

    Stratified by **class only**, deliberately. The background is a reference
    distribution - "what does this model see normally?" - not an evaluation
    sample, so it does not need every training run represented, and
    stratifying it by (`split_group`, `class`) as well imposes a floor of one
    row per stratum: ~1,000 rows on the 212-run train partitions here, when
    100 were asked for.

    That floor is not cosmetic. Interventional SHAP costs O(rows x background),
    so a background ten times larger than requested makes the card ten times
    more expensive, and it was measured doing exactly that (§3 of the card:
    411 rows returned for a cap of 100 on a 265-run smoke). Classes are kept
    because a background missing a class entirely would give that class's
    attributions a reference the model never sees.
    """
    import numpy as np

    if not cap or len(positions) <= cap:
        return positions
    rng = np.random.RandomState(seed)
    labels = y[positions]
    present = np.unique(labels)
    # One row per class first, so no class can be lost to rounding, then the
    # remainder proportionally.
    chosen = []
    remaining = cap - len(present)
    for label in present:
        pool = positions[labels == label]
        share = len(pool) / len(positions)
        take = 1 + int(round(max(remaining, 0) * share))
        take = min(take, len(pool))
        chosen.append(rng.choice(pool, size=take, replace=False))
    sample = np.unique(np.concatenate(chosen))
    if len(sample) > cap:
        sample = np.sort(rng.choice(sample, size=cap, replace=False))
    return sample


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------

def explain_fold(model, X, y, explain_positions, background_positions, classes,
                 feature_names):
    """Mean |SHAP| per feature per class, on held-out rows.

    The explainer is built **with** the background and with
    `feature_perturbation="interventional"`; `tree_path_dependent` would drop
    the background silently and answer a different question, so the mode is
    asserted after construction rather than trusted.
    """
    import numpy as np
    import shap

    background = np.ascontiguousarray(X[background_positions])
    explainer = shap.TreeExplainer(
        model, data=background, feature_perturbation="interventional")
    mode = getattr(explainer, "feature_perturbation", None)
    if mode != "interventional":
        raise ShapRunError(
            "shap fell back to %r: the background from the train partition "
            "would be ignored, and F.3 asks for it explicitly" % mode)

    rows = np.ascontiguousarray(X[explain_positions])
    values = explainer.shap_values(rows, check_additivity=False)

    # shap returns either a list of per-class arrays or one stacked array with
    # the class on the last axis, depending on version and model. Both are
    # normalised to (n_classes, n_rows, n_features) here so the report does
    # not depend on which shape came back.
    if isinstance(values, list):
        stacked = np.stack([np.asarray(v) for v in values], axis=0)
    else:
        values = np.asarray(values)
        if values.ndim != 3:
            raise ShapRunError(
                "expected per-class SHAP values, got an array of shape %s"
                % (values.shape,))
        stacked = np.transpose(values, (2, 0, 1))

    if stacked.shape[0] != len(classes):
        raise ShapRunError(
            "SHAP returned %d class blocks for %d classes"
            % (stacked.shape[0], len(classes)))
    if stacked.shape[2] != len(feature_names):
        raise ShapRunError(
            "SHAP returned %d features for %d model columns"
            % (stacked.shape[2], len(feature_names)))

    magnitude = np.abs(stacked)
    del stacked

    def block(mean_abs):
        return {
            str(classes[index]): {
                name: float(mean_abs[index, position])
                for position, name in enumerate(feature_names)
            }
            for index in range(len(classes))
        }

    # Global: every explained row, which on this pool is ~95.6% `normal`.
    result = {"global": block(magnitude.mean(axis=1)), "conditional": {},
              "support": {}}

    # Conditional: rows whose *true* class is the one named. A class with no
    # row in this fold's sample gets no entry rather than a zero, because a
    # zero would rank as "this feature does nothing" instead of "nothing was
    # measured".
    true_labels = y[explain_positions]
    for index, class_name in enumerate(classes):
        selected = np.flatnonzero(true_labels == index)
        result["support"][str(class_name)] = int(len(selected))
        if len(selected) == 0:
            continue
        result["conditional"][str(class_name)] = block(
            magnitude[:, selected, :].mean(axis=1))

    del magnitude, rows, background, explainer
    gc.collect()
    return result


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def stability(per_fold_values):
    """Spread of one feature's importance across the folds (F.4).

    Reported as min/median/max plus the ratio of max to median rather than as
    an interval: five folds are one partition of the same 265 runs, not five
    draws from the generator, and `bootstrap_run_intervals.py` is explicit
    that dressing that up as an error bar overstates it.
    """
    import numpy as np

    values = np.asarray(per_fold_values, dtype=float)
    median = float(np.median(values))
    return {
        "folds": [float(v) for v in values],
        "min": float(values.min()),
        "median": median,
        "max": float(values.max()),
        "max_over_median": float(values.max() / median) if median > 0 else None,
    }


def aggregate(fold_blocks, classes, feature_names):
    """Fold-to-fold spread per output class per feature (F.4)."""
    return {
        str(class_name): {
            feature: stability([block[str(class_name)][feature]
                                for block in fold_blocks])
            for feature in feature_names
        }
        for class_name in classes
    }


def aggregate_conditional(fold_importances, classes, feature_names):
    """The same, restricted to rows of each true class.

    A fold where a true class had no sampled row contributes nothing rather
    than a zero, and the number of folds that did contribute travels with the
    values - a ranking from two folds is not a ranking from five.
    """
    out = {}
    for true_name in classes:
        blocks = [fold["conditional"][str(true_name)] for fold in fold_importances
                  if str(true_name) in fold["conditional"]]
        rows = sum(fold["support"].get(str(true_name), 0)
                   for fold in fold_importances)
        if not blocks:
            out[str(true_name)] = {"folds_measured": 0, "rows": rows,
                                   "per_output_class": {}}
            continue
        out[str(true_name)] = {
            "folds_measured": len(blocks),
            "rows": rows,
            "per_output_class": aggregate(blocks, classes, feature_names),
        }
    return out


def build_report(payload, top_n=12):
    lines = [
        "# SHAP on held-out grouped data (card F.3/F.4)",
        "",
        "- Generated: %s" % payload["generated"],
        "- Reference run explained: `%s` (model `%s`, balance `%s`)"
        % (payload["reference_run"], payload["model"], payload["balance"]),
        "- Dataset `%s…`, protocol `%s`, %d folds from `%s`"
        % (str(payload["dataset_sha256"])[:12], payload["protocol"],
           len(payload["folds"]), os.path.basename(str(payload["splits"]))),
        "- Explained rows per fold: **%s** (held-out test groups), background "
        "rows per fold: **%s** (train groups of the same fold)"
        % (format(payload["explain_rows_per_fold"], ","),
           format(payload["background_rows_per_fold"], ",")),
        "- Perturbation: `%s`" % payload["feature_perturbation"],
        "",
    ]

    if payload["verified"]:
        lines += [
            "**The model explained is the model published.** Each fold was "
            "refit from the persisted split at the reference run's own seed "
            "and thread count, and its predictions reproduce "
            "`%s` row for row - %s rows checked across %d folds, 0 "
            "mismatches. Without that, a SHAP value here would explain a "
            "model the paper does not report."
            % (PREDICTIONS_NAME, format(payload["rows_verified"], ","),
               len(payload["folds"])),
            "",
        ]
    else:
        lines += [
            "> **UNVERIFIED.** The refit was not checked against the reference "
            "run's predictions (`--skip-verification`), so nothing here is "
            "established to explain the published model.",
            "",
        ]

    lines += [
        "SHAP attributes *this model's output* to *its inputs*. It is not a "
        "statement about what causes a grayhole, or about what happens in a "
        "substation (`explainability_card.md` §7). The label is per-message, "
        "which bounds every explanation exactly as it bounds every recall "
        "number (`label_duplication_audit.md` §7).",
        "",
        "Values are **mean |SHAP| per feature, per class, per fold**, never "
        "summed across classes: a feature that separates two attack families "
        "and a feature that separates attacks from `normal` are different "
        "findings.",
        "",
    ]

    lines += [
        "## Global importance: every held-out row",
        "",
        "Mean |SHAP| over the whole explained sample, which is a proportional "
        "replica of the held-out distribution and therefore **~95.6% "
        "`normal`**. This answers \"what moves this class's output across "
        "held-out traffic\", and on a pool at 2% attack prevalence that is "
        "largely a statement about normal messages. For \"what the model "
        "keys on when it is looking at an actual attack of this family\", "
        "read the conditional section below instead.",
        "",
    ]

    def table(features, heading):
        ordered = sorted(features.items(), key=lambda kv: -kv[1]["median"])
        rows = [
            heading,
            "",
            "| Rank | Feature | median mean\\|SHAP\\| | min | max | max/median |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for rank, (name, block) in enumerate(ordered[:top_n], start=1):
            ratio = block["max_over_median"]
            rows.append("| %d | `%s` | %.6g | %.6g | %.6g | %s |" % (
                rank, name, block["median"], block["min"], block["max"],
                "—" if ratio is None else "%.2f" % ratio))
        rows.append("")
        return rows

    for class_name, features in payload["importances"].items():
        lines += table(features, "### `%s`" % class_name)

    lines += [
        "## Conditional importance: rows of that class only",
        "",
        "Mean |SHAP| for a class's own output, over the held-out rows whose "
        "**true** class is that class. This is the question an explainability "
        "claim in this paper makes, and it is not recoverable from the global "
        "average above.",
        "",
        "It also rests on far fewer rows, so each table says how many and "
        "across how many folds. An attack class here carries ~90-120 rows per "
        "fold from ~9 runs; a ranking that is not stable across the folds is "
        "not a finding.",
        "",
    ]
    for true_name, block in payload["conditional_importances"].items():
        if not block["per_output_class"]:
            lines += ["### `%s`" % true_name, "",
                      "Not measured: no held-out row of this class was sampled.",
                      ""]
            continue
        own = block["per_output_class"].get(true_name)
        lines += table(
            own,
            "### `%s` — %s rows across %d of %d folds"
            % (true_name, format(block["rows"], ","), block["folds_measured"],
               len(payload["folds"])))

    lines += [
        "## How to read the spread",
        "",
        "`min`/`max` are across the five folds. They are **not** a confidence "
        "interval: the folds are one partition of the same 265 runs, not five "
        "draws from the generator. A class carried by ~9 runs per test fold "
        "moves in steps of roughly one run's worth of attribution, which is "
        "the same floor `bootstrap_run_intervals.py` reports for its metrics.",
        "",
    ]
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--reference-run", required=True,
                        help="The published run whose model is explained. Its "
                             "predictions are what the refit must reproduce.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-folds", type=int, default=0,
                        help="Run only the first N folds. For pricing the card "
                             "before committing to all five; the output is "
                             "marked as a partial probe and is not a result.")
    parser.add_argument("--explain-max-rows", type=int, default=20_000,
                        help="Held-out rows explained per fold, sampled "
                             "proportionally inside every (split_group, class) "
                             "stratum so all test runs stay represented. 0 "
                             "explains the whole test partition. The default "
                             "is measured, not guessed: see "
                             "explainability_card.md §3.")
    parser.add_argument("--background-rows", type=int, default=100,
                        help="Train rows per fold forming the background "
                             "distribution, stratified by class only. "
                             "Interventional SHAP is linear in this, so it is "
                             "small by necessity; the bias it leaves flattens "
                             "differences rather than inventing them "
                             "(explainability_card.md §3).")
    parser.add_argument("--skip-verification", action="store_true",
                        help="Do not check the refit against the reference "
                             "predictions. Marks the output UNVERIFIED "
                             "everywhere. For a reference run whose "
                             "predictions are unavailable, not for a hurry.")
    parser.add_argument("--top-n", type=int, default=12,
                        help="Features listed per class in the Markdown "
                             "report. The JSON always carries all of them.")
    # Passed through so the refit is configured exactly like the run it
    # reproduces; the defaults are the champion's.
    parser.add_argument("--model", default="xgboost")
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--feature-set", default="all")
    parser.add_argument("--discard-column", action="append", default=[])
    parser.add_argument("--max-train-rows-per-fold", type=int, default=0)
    parser.add_argument("--max-rows-per-group-class", type=int, default=0,
                        help="Non-zero enables a technical smoke sample.")
    parser.add_argument("--balance", choices=["none"], default="none",
                        help="Card F explains the published detector, which "
                             "is the unbalanced one (§19).")
    parser.add_argument("--smote-oversample-factor", type=float, default=20.0)
    parser.add_argument("--smote-max-target", type=int, default=200_000)
    # `provenance_block` records it; this card never writes posteriors.
    parser.set_defaults(save_scores=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        import numpy as np

        inputs = load_inputs(args)
        arrays = inputs["arrays"]
        splits = inputs["splits"]["splits"]
        classes = list(arrays["classes"])
        feature_names = list(arrays["features"])
        X = arrays["X"]
        y = arrays["y"]
        group_codes = arrays["group_codes"]
        row_index = arrays["row_index"]

        reference = load_reference(args.reference_run, inputs["digest"], args.splits)
        if reference.get("model") != args.model:
            raise ShapRunError(
                "reference run is model %r but the refit is configured for %r"
                % (reference.get("model"), args.model))
        if reference.get("seed") != args.seed:
            raise ShapRunError(
                "reference run used seed %r, the refit %r - the model would "
                "differ" % (reference.get("seed"), args.seed))
        # `feature_set` is recorded as None by runs made before D.1 added the
        # flag, and those runs are the full feature set - which is what
        # `all` means. Anything else must match exactly: explaining a model
        # fitted without a column group, with a refit that has it, is not a
        # near miss but a different model.
        reference_features = reference.get("feature_set") or "all"
        if reference_features != args.feature_set:
            raise ShapRunError(
                "reference run was fitted on feature set %r but the refit is "
                "configured for %r - it would be a different model"
                % (reference_features, args.feature_set))
        if reference.get("n_jobs") != args.n_jobs:
            raise ShapRunError(
                "reference run used n_jobs=%r, the refit %r. It is a "
                "throughput knob, but XGBoost's histogram reduction order "
                "depends on it, so the refit could differ in the last "
                "decimals and fail verification for the wrong reason."
                % (reference.get("n_jobs"), args.n_jobs))

        if args.max_rows_per_group_class and not args.skip_verification:
            raise ShapRunError(
                "a capped technical smoke reads a sample of the dataset, so "
                "its refit cannot reproduce the reference run's predictions "
                "and verification would fail for the wrong reason. Pass "
                "--skip-verification to run the smoke, and expect the output "
                "to be marked UNVERIFIED.")

        expected = None if args.skip_verification else reference_predictions(
            args.reference_run, classes)

        os.makedirs(args.out_dir, exist_ok=True)
        fold_importances = []
        fold_records = []
        rows_verified = 0

        if args.limit_folds:
            splits = splits[:args.limit_folds]

        for fold_index, split in enumerate(splits):
            started = time.time()
            split_id = split["split_id"]
            train_positions, test_positions = fold_positions(
                arrays, split, args.seed, fold_index,
                args.max_train_rows_per_fold)

            model = classifier(args.model, args.seed + fold_index,
                               n_jobs=args.n_jobs)
            model.fit(X[train_positions], y[train_positions])
            print("  %s refit on %s rows (%.1f min)"
                  % (split_id, format(len(train_positions), ","),
                     (time.time() - started) / 60.0), flush=True)

            if expected is not None:
                predicted = predict_in_chunks(model, X, test_positions)
                rows_verified += verify_fold(
                    split_id, row_index, test_positions, predicted, reference=expected)
                del predicted
                gc.collect()
                print("    predictions reproduce the reference run", flush=True)

            explain_positions = stratified_sample(
                test_positions, group_codes, y, len(classes),
                args.explain_max_rows, args.seed + fold_index)
            background_positions = background_sample(
                train_positions, y, args.background_rows, args.seed + fold_index)

            importances = explain_fold(
                model, X, y, explain_positions, background_positions,
                classes, feature_names)
            fold_importances.append(importances)
            fold_records.append({
                "split_id": split_id,
                "train_rows": int(len(train_positions)),
                "test_rows": int(len(test_positions)),
                "explained_rows": int(len(explain_positions)),
                "background_rows": int(len(background_positions)),
                "explained_groups": int(len(np.unique(group_codes[explain_positions]))),
                "explained_support": importances["support"],
                "seconds": time.time() - started,
            })
            print("  %s explained %s held-out rows against %s background rows "
                  "(%.1f min)"
                  % (split_id, format(len(explain_positions), ","),
                     format(len(background_positions), ","),
                     (time.time() - started) / 60.0), flush=True)
            del model, train_positions, test_positions
            del explain_positions, background_positions
            gc.collect()

        payload = provenance_block(args, inputs, classes, feature_names)
        payload.update({
            "card": "F.3/F.4",
            "partial_probe": bool(args.limit_folds),
            "reference_run": os.path.abspath(args.reference_run),
            "feature_perturbation": "interventional",
            "verified": expected is not None,
            "rows_verified": rows_verified,
            "explain_rows_per_fold": args.explain_max_rows,
            "background_rows_per_fold": args.background_rows,
            "folds": fold_records,
            "importances": aggregate(
                [fold["global"] for fold in fold_importances],
                classes, feature_names),
            "conditional_importances": aggregate_conditional(
                fold_importances, classes, feature_names),
            "per_fold_importances": fold_importances,
        })
        payload["generated"] = datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC")

        with open(os.path.join(args.out_dir, IMPORTANCES_NAME), "w",
                  encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        with open(os.path.join(args.out_dir, "shap_importances.md"), "w",
                  encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(build_report(payload, args.top_n)).rstrip("\n"))
            handle.write("\n")
    except (OSError, json.JSONDecodeError, GroupedRunError, ShapRunError,
            ValueError, KeyError, TypeError) as exc:
        print("GROUPED SHAP FAILED\n%s" % exc, file=sys.stderr)
        return 1

    print("SHAP importances written: %s" % os.path.join(args.out_dir, IMPORTANCES_NAME))
    print("  %d folds, %s held-out rows explained in total"
          % (len(fold_records),
             format(sum(f["explained_rows"] for f in fold_records), ",")))
    if payload["verified"]:
        print("  refit verified against %s: %s rows, 0 mismatches"
              % (os.path.basename(args.reference_run), format(rows_verified, ",")))
    else:
        print("  UNVERIFIED: the refit was not checked against the reference run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
