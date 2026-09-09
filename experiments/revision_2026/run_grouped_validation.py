"""Run a model strictly from persisted grouped splits.

Checklist B.1/B.3/B.5.  This is the canonical revision runner: it verifies the
prepared-dataset audit and split-file hash, never creates a message-level split,
and records one prediction with its ``split_group`` and fold identifier.

Use ``--max-rows-per-group-class`` for a fast technical smoke test.  A capped
run validates wiring only and must not be reported as scientific performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


IDENTIFIER_COLUMNS = {
    "run_id", "trace_id", "event_id", "scenario_id", "seed",
    "attack_variant", "loss_rate", "burst_size", "traffic_rate",
    "substation_config", "split_group", "batch_index",
    "message_index",
    # Card-C benign-degradation provenance. impairment_mode is a trivial leak
    # if it ever reached the feature matrix - it directly encodes whether a
    # row is `benign_degradation` (see benign_controls.md SS6). The other two
    # are its run-level config, discarded for the same reason loss_rate/
    # burst_size are.
    "impairment_mode", "impairment_rate", "impairment_intensity_ms",
}
BASE_DISCARD_COLUMNS = {
    "ethDst", "ethSrc", "gocbRef", "datSet", "goID", "test", "ndsCom",
    "protocol", "ethType", "TPID", "gooseAppid",
}


class GroupedRunError(ValueError):
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


def verify_artifacts(dataset, preparation_report, splits_payload):
    dataset_abs = os.path.abspath(dataset)
    digest = sha256_file(dataset)
    if preparation_report.get("status") != "pass":
        raise GroupedRunError("preparation report did not pass")
    if os.path.abspath(preparation_report.get("output", "")) != dataset_abs:
        raise GroupedRunError("preparation report belongs to a different dataset")
    if preparation_report.get("output_sha256") != digest:
        raise GroupedRunError("prepared dataset hash differs from preparation report")
    if splits_payload.get("dataset_sha256") != digest:
        raise GroupedRunError("split file belongs to a different dataset version")
    if splits_payload.get("open_set_diagnostic"):
        raise GroupedRunError(
            "open-set LOETO splits are diagnostic and cannot be used for standard multiclass metrics"
        )
    return digest


def load_frame(path):
    import pandas as pd
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    raise GroupedRunError("dataset must be .csv or .parquet")


def technical_sample(frame, group_column, target_column, cap, seed):
    if not cap:
        return frame
    # Sampling happens independently inside group x class strata. It is only a
    # speed control; the report is marked technical_smoke and never used as a
    # substitute for original-distribution evaluation.
    shuffled = frame.sample(frac=1, random_state=seed)
    sampled = (
        shuffled.groupby([group_column, target_column], group_keys=False, observed=True)
        .head(cap)
        .sort_index()
    )
    return sampled


def load_technical_sample(path, group_column, target_column, cap, seed):
    """Read a capped, per-(group, class) sample without loading the dataset.

    Every `split_group` value this pipeline produces - whatever
    `--split-level` it was derived at - lives entirely inside one Parquet
    row group: `add_experiment_metadata.py`'s native path and
    `prepare_grouped_dataset.py` both write one row group per run, a run is
    one trace, and an event never spans two traces. So every (group, class)
    stratum `technical_sample` draws from is complete within a single row
    group, and sampling row group by row group - instead of loading
    everything and shuffling once - draws from the exact same eligible pool
    per stratum and respects the same `cap`, without ever materialising the
    full (possibly tens-of-millions-of-row) table just to throw most of it
    away. It does not reproduce the identical specific rows a single
    whole-dataset shuffle with the same seed would have picked (each chunk
    is shuffled independently - see the per-chunk seed offset below); that
    was never a documented guarantee of a technical smoke sample, only the
    cap and coverage are. Only called for a `.parquet` dataset; see `main()`.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    samples = []
    row_offset = 0
    for index in range(parquet_file.num_row_groups):
        chunk = parquet_file.read_row_group(index).to_pandas()
        # `to_pandas()` restarts the index at 0 per row group; re-anchoring
        # it to the row's true position in the full prepared dataset keeps
        # `row_index` values recorded downstream (predictions,
        # `benign_confusion_report.py`'s positional rejoin) valid.
        chunk.index = pd.RangeIndex(row_offset, row_offset + len(chunk))
        row_offset += len(chunk)
        # Offset per chunk (same pattern as `classifier(model_name, seed +
        # fold_index)` below) so same-sized chunks don't all draw the same
        # relative shuffle from a shared seed - each group/class stratum
        # still gets exactly `cap` rows, just not the same specific ones a
        # single whole-dataset shuffle would have picked.
        sampled_chunk = technical_sample(chunk, group_column, target_column, cap, seed + index)
        if len(sampled_chunk):
            samples.append(sampled_chunk)
    if not samples:
        return pd.DataFrame(columns=[])
    return pd.concat(samples).sort_index()


def feature_matrix(frame, target_column, extra_discard):
    discard = IDENTIFIER_COLUMNS | BASE_DISCARD_COLUMNS | {target_column} | set(extra_discard)
    features = frame.drop(columns=[c for c in discard if c in frame.columns])
    non_numeric = list(features.select_dtypes(exclude="number").columns)
    if non_numeric:
        raise GroupedRunError(
            "non-numeric feature columns remain; discard or encode them explicitly: %s" % non_numeric
        )
    if features.empty:
        raise GroupedRunError("no model features remain")
    return features


def classifier(name, seed):
    if name == "decision-tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=8, random_state=seed)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", random_state=seed
        )
    raise GroupedRunError("unknown model: %s" % name)


def run_folds(frame, splits, group_column, target_column, model_name, seed,
              extra_discard, predictions_writer):
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[target_column].astype(str))
    X = feature_matrix(frame, target_column, extra_discard)
    groups = frame[group_column].astype(str)
    all_labels = list(range(len(encoder.classes_)))
    metrics = []

    for fold_index, split in enumerate(splits):
        train_mask = groups.isin(split["train_groups"]).to_numpy()
        test_mask = groups.isin(split["test_groups"]).to_numpy()
        if (train_mask & test_mask).any():
            raise GroupedRunError("%s has row-level train/test overlap" % split["split_id"])
        if not train_mask.any() or not test_mask.any():
            raise GroupedRunError("%s has an empty partition" % split["split_id"])

        train_classes = set(y[train_mask])
        test_classes = set(y[test_mask])
        if not test_classes.issubset(train_classes):
            missing = encoder.inverse_transform(sorted(test_classes - train_classes)).tolist()
            raise GroupedRunError("%s test-only classes: %s" % (split["split_id"], missing))

        model = classifier(model_name, seed + fold_index)
        model.fit(X.loc[train_mask], y[train_mask])
        predicted = model.predict(X.loc[test_mask])
        report = classification_report(
            y[test_mask], predicted, labels=all_labels,
            target_names=encoder.classes_, output_dict=True, zero_division=0,
        )
        metrics.append({
            "split_id": split["split_id"],
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "accuracy": float(accuracy_score(y[test_mask], predicted)),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "per_class": {
                label: {
                    key: float(report[label][key])
                    for key in ("precision", "recall", "f1-score", "support")
                }
                for label in encoder.classes_
            },
        })
        test_indices = frame.index[test_mask]
        true_labels = encoder.inverse_transform(y[test_mask])
        predicted_labels = encoder.inverse_transform(np.asarray(predicted, dtype=int))
        # Written straight to disk instead of accumulated in a list: on a
        # full run every dataset row is a test row in exactly one fold, so
        # the list would otherwise hold one dict per row for the whole run.
        for row_index, group, truth, prediction in zip(
                test_indices, groups.loc[test_mask], true_labels, predicted_labels):
            predictions_writer.writerow({
                "split_id": split["split_id"], "row_index": int(row_index),
                "split_group": group, "y_true": truth, "y_pred": prediction,
            })
    return metrics, list(encoder.classes_), list(X.columns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--preparation-report", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", choices=["decision-tree", "xgboost"], default="decision-tree")
    parser.add_argument("--group-column", default="split_group")
    parser.add_argument("--target-column", default="class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows-per-group-class", type=int, default=0,
                        help="Non-zero enables a technical smoke sample.")
    parser.add_argument("--discard-column", action="append", default=[])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        preparation = load_json(args.preparation_report)
        split_payload = load_json(args.splits)
        digest = verify_artifacts(args.dataset, preparation, split_payload)

        if args.max_rows_per_group_class and args.dataset.lower().endswith((".parquet", ".pq")):
            # A capped (technical smoke) run only ever needs a small sample,
            # so it is read chunked - see load_technical_sample - and the
            # full dataset is never loaded. Column existence is checked from
            # the file's schema alone, before any row is read.
            import pyarrow.parquet as pq
            columns = pq.ParquetFile(args.dataset).schema_arrow.names
            if args.group_column not in columns or args.target_column not in columns:
                raise GroupedRunError("dataset is missing group or target column")
            frame = load_technical_sample(
                args.dataset, args.group_column, args.target_column,
                args.max_rows_per_group_class, args.seed,
            )
        else:
            frame = load_frame(args.dataset)
            if args.group_column not in frame or args.target_column not in frame:
                raise GroupedRunError("dataset is missing group or target column")
            frame = technical_sample(
                frame, args.group_column, args.target_column,
                args.max_rows_per_group_class, args.seed,
            )
        os.makedirs(args.out_dir, exist_ok=True)
        predictions_path = os.path.join(args.out_dir, "grouped_predictions.csv")
        # Predictions are written straight to disk as each fold finishes
        # instead of being collected into one Python list first (on a full
        # run that list would hold one dict per dataset row). Staged under
        # `.tmp` and renamed into place only once the whole run - including
        # the JSON report - has succeeded, so a failure partway through a
        # fold still leaves no partial `grouped_predictions.csv` behind,
        # matching the previous all-or-nothing behaviour.
        predictions_tmp = predictions_path + ".tmp"
        with open(predictions_tmp, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["split_id", "row_index", "split_group", "y_true", "y_pred"]
            )
            writer.writeheader()
            metrics, classes, features = run_folds(
                frame, split_payload["splits"], args.group_column, args.target_column,
                args.model, args.seed, args.discard_column, writer,
            )
        report = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "status": "technical_smoke" if args.max_rows_per_group_class else "full_grouped_run",
            "dataset": os.path.abspath(args.dataset),
            "dataset_sha256": digest,
            "splits": os.path.abspath(args.splits),
            "protocol": split_payload["protocol"],
            "model": args.model,
            "seed": args.seed,
            "sample_cap_per_group_class": args.max_rows_per_group_class or None,
            "rows_used": len(frame),
            "classes": classes,
            "features": features,
            "fold_metrics": metrics,
        }
        with open(os.path.join(args.out_dir, "grouped_validation_report.json"),
                  "w", encoding="utf-8", newline="\n") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
        os.replace(predictions_tmp, predictions_path)
    except (OSError, json.JSONDecodeError, GroupedRunError, ValueError) as exc:
        try:
            if os.path.exists(predictions_tmp):
                os.remove(predictions_tmp)
        except (OSError, NameError):
            pass
        print("GROUPED VALIDATION FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print("Completed %d grouped folds on %d rows (%s)." %
          (len(metrics), len(frame), report["status"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
