"""Prepare Gray-GOOSE data for leakage-free grouped validation.

Checklist B.2: inter-message delta features are recomputed only after sorting
inside each trace.  The first retained row of every trace is dropped because
its predecessor is not present; filling a synthetic zero would invent protocol
behaviour.  ``T-UNRESOLVED`` is rejected because it mixes several real traces.

The formulas reproduce ERENO's ``IntermessageCorrelation`` implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


DELTA_SOURCES = {
    "stDiff": "StNum",
    "sqDiff": "SqNum",
    "gooseLengthDiff": "gooseLen",
    "apduSizeDiff": "APDUSize",
    "frameLengthDiff": "frameLen",
    "timestampDiff": "GooseTimestamp",
    "tDiff": "t",
}
DELTA_COLUMNS = list(DELTA_SOURCES) + ["cbStatusDiff", "timeFromLastChange"]
FORBIDDEN_TRACES = {"T-UNRESOLVED"}


class PreparationError(ValueError):
    pass


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute_trace_deltas(df, trace_column="trace_id", batch_column="batch_index",
                           first_row_policy="drop"):
    """Return rows ordered by trace/time with leakage-safe deltas.

    ``first_row_policy`` can be ``drop`` (recommended) or ``nan``.  Zero is
    deliberately unsupported: zero means "no change", which is not known at a
    trace boundary.
    """
    import numpy as np
    import pandas as pd

    required = {trace_column, batch_column, "cbStatus", "t"} | set(DELTA_SOURCES.values())
    missing = sorted(required - set(df.columns))
    if missing:
        raise PreparationError("dataset is missing columns: %s" % missing)
    if first_row_policy not in ("drop", "nan"):
        raise PreparationError("first_row_policy must be `drop` or `nan`")
    if df[trace_column].isna().any():
        raise PreparationError("trace identifiers contain null values")

    trace_values = set(df[trace_column].astype(str).unique())
    forbidden = sorted(trace_values & FORBIDDEN_TRACES)
    if forbidden:
        raise PreparationError(
            "cannot recompute deltas for mixed/non-independent traces: %s" % forbidden
        )

    out = df.copy()
    marker = "__revision_original_order__"
    if marker in out.columns:
        raise PreparationError("reserved column already exists: %s" % marker)
    out[marker] = np.arange(len(out), dtype=np.int64)

    # `GooseTimestamp`, StNum and SqNum are protocol values, not safe ordering
    # keys: attacks can make them non-monotonic. The native writer's row order
    # is the actual sequence used by IntermessageCorrelation. batch_index plus
    # stable source order preserves that sequence after runs are pooled.
    sort_columns = [trace_column, batch_column, marker]
    out = out.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    grouped = out.groupby(trace_column, sort=False, observed=True)
    out["message_index"] = grouped.cumcount()
    first = out["message_index"].eq(0)

    for destination, source in DELTA_SOURCES.items():
        numeric = pd.to_numeric(out[source], errors="raise")
        out[destination] = numeric.groupby(out[trace_column], sort=False).diff()

    status = out["cbStatus"]
    previous_status = grouped["cbStatus"].shift(1)
    out["cbStatusDiff"] = status.ne(previous_status).astype(float)
    out.loc[first, "cbStatusDiff"] = np.nan

    timestamp = pd.to_numeric(out["GooseTimestamp"], errors="raise")
    event_time = pd.to_numeric(out["t"], errors="raise")
    out["timeFromLastChange"] = timestamp - event_time

    intermessage = [c for c in DELTA_COLUMNS if c != "timeFromLastChange"]
    if not out.loc[first, intermessage].isna().all().all():
        raise AssertionError("a delta crossed a trace boundary")
    rows_before = len(out)
    boundary_rows = int(first.sum())
    if first_row_policy == "drop":
        # `.loc[boolean mask]` already returns an independent copy, so an
        # explicit `.copy()` here would just duplicate the same allocation.
        out = out.loc[~first]
    out = out.drop(columns=[marker]).reset_index(drop=True)

    audit = {
        "rows_before": rows_before,
        "rows_after": len(out),
        "traces": len(trace_values),
        "boundary_rows": boundary_rows,
        "first_row_policy": first_row_policy,
        "trace_column": trace_column,
        "ordering": [trace_column, batch_column, "stable source row order"],
        "delta_columns": DELTA_COLUMNS,
    }
    return out, audit


def load_frame(path):
    import pandas as pd
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, encoding="utf-8")
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    raise PreparationError("input must be .csv or .parquet")


def write_frame(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    elif path.lower().endswith((".parquet", ".pq")):
        df.to_parquet(path, index=False)
    else:
        raise PreparationError("output must be .csv or .parquet")


def iter_row_group_frames(path):
    """Yield the dataset one Parquet row group at a time.

    `merge_runs.py` and `add_experiment_metadata.py`'s native (chunked) path
    each write exactly one row group per run, and a run is exactly one trace
    - so *when the file was written that way*, a row group already holds
    everything `recompute_trace_deltas` needs for one trace, and processing
    it in isolation is identical to processing it as part of one big frame.

    This is NOT true of every Parquet file, though: a plain
    ``DataFrame.to_parquet()`` call chunks purely by row count (pyarrow's
    default is 2**20 rows), with no regard for trace boundaries, so a trace
    can straddle two row groups. ``main()`` below defends against exactly
    that by rejecting a trace id it has already seen in an earlier chunk
    instead of silently computing wrong deltas for the split trace's tail.
    """
    import pandas as pd
    if path.lower().endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(path)
        for index in range(parquet_file.num_row_groups):
            yield parquet_file.read_row_group(index).to_pandas()
    elif path.lower().endswith(".csv"):
        yield pd.read_csv(path, encoding="utf-8")
    else:
        raise PreparationError("input must be .csv or .parquet")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace-column", default="trace_id")
    parser.add_argument("--batch-column", default="batch_index")
    parser.add_argument("--first-row-policy", choices=["drop", "nan"], default="drop")
    parser.add_argument("--report", help="JSON preparation audit path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    writer = None
    csv_header_written = False
    # Written under `.tmp` and renamed into place only once every chunk has
    # succeeded, so a failure partway through the dataset still leaves no
    # partial output file behind - matching the previous all-or-nothing
    # behaviour, when the whole dataset was one in-memory frame.
    out_tmp = args.out + ".tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        rows_before = 0
        rows_after = 0
        boundary_rows = 0
        trace_values = set()

        for chunk in iter_row_group_frames(args.dataset):
            # A trace split across two row groups would otherwise get a
            # spurious extra "first row" per fragment and silently wrong
            # deltas at the split point - caught here, before that can
            # happen, rather than trusting every possible Parquet writer to
            # align row groups to traces (see iter_row_group_frames).
            chunk_trace_values = set(chunk[args.trace_column].astype(str).unique())
            repeated = chunk_trace_values & trace_values
            if repeated:
                raise PreparationError(
                    "trace(s) %s span more than one row group - this file's row "
                    "groups are not trace-aligned, so chunked delta recomputation "
                    "cannot run on it safely. Regenerate it via "
                    "add_experiment_metadata.py's native (chunked) path, which "
                    "writes one row group per run/trace." % sorted(repeated)
                )
            trace_values.update(chunk_trace_values)
            prepared_chunk, chunk_audit = recompute_trace_deltas(
                chunk, args.trace_column, args.batch_column, args.first_row_policy
            )
            rows_before += chunk_audit["rows_before"]
            rows_after += chunk_audit["rows_after"]
            boundary_rows += chunk_audit["boundary_rows"]

            if args.out.lower().endswith((".parquet", ".pq")):
                import pyarrow as pa
                import pyarrow.parquet as pq
                table = pa.Table.from_pandas(prepared_chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_tmp, table.schema)
                writer.write_table(table)
            elif args.out.lower().endswith(".csv"):
                prepared_chunk.to_csv(
                    out_tmp, index=False,
                    mode="w" if not csv_header_written else "a",
                    header=not csv_header_written,
                )
                csv_header_written = True
            else:
                raise PreparationError("output must be .csv or .parquet")
        if writer is not None:
            writer.close()
            writer = None

        audit = {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "traces": len(trace_values),
            "boundary_rows": boundary_rows,
            "first_row_policy": args.first_row_policy,
            "trace_column": args.trace_column,
            "ordering": [args.trace_column, args.batch_column, "stable source row order"],
            "delta_columns": DELTA_COLUMNS,
            "status": "pass",
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "source": os.path.abspath(args.dataset),
            "output": os.path.abspath(args.out),
            "output_sha256": sha256_file(out_tmp),
        }
        os.replace(out_tmp, args.out)
        report = args.report or os.path.splitext(args.out)[0] + ".preparation.json"
        with open(report, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(audit, fh, indent=2)
            fh.write("\n")
    except (OSError, PreparationError, ValueError) as exc:
        if writer is not None:
            writer.close()
        if os.path.exists(out_tmp):
            os.remove(out_tmp)
        print("GROUPED DATA PREPARATION FAILED\n%s" % exc, file=sys.stderr)
        return 1
    print("Prepared %d rows across %d traces; removed %d boundary rows." %
          (audit["rows_after"], audit["traces"],
           audit["rows_before"] - audit["rows_after"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
