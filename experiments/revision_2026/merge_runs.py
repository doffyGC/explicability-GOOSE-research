"""
Pool the per-run CSVs produced by `generate_run_matrix.py` into one dataset.

Checklist item A.3, second half. `generate_run_matrix.py` drives the patched
ERENO once per cell of the run matrix and leaves one CSV per run, each carrying
its own provenance columns (`RunContext.csvHeader`). This script concatenates
them into a single table and, more importantly, *checks that the runs are what
they claim to be* before they are pooled.

The checks exist because the matrix script configures ERENO by rewriting a
single global file (`params.properties`) and restoring it afterwards. That
works, but it means an interrupted or hand-edited run can leave a CSV whose
contents disagree with its filename, and a mislabelled run silently corrupts
grouped cross-validation: the group identity is exactly what the split relies
on. Every check below is designed to fail loudly rather than pool bad data.

  header consistency  - all runs share one schema, so pooling cannot misalign
                        columns
  self-consistency    - the run_id/seed inside the rows match the sidecar and
                        the filename (catches a run generated with a stale
                        params.properties)
  single run per file - exactly one run_id per CSV (catches a file that got
                        appended to instead of overwritten)
  distinct identity   - no run_id appears twice across files
  distinct content    - no two runs have identical payloads. Two runs that
                        differ only by their seed column but carry the same
                        messages are not two experimental units, and pooling
                        them would inflate the apparent number of independent
                        groups. This is the check that would have caught the
                        unseeded generator the revision had to fix.

What this script does NOT do
----------------------------
It does not derive `event_id` or `split_group`, and it does not touch
`attack_variant`. Those belong to `add_experiment_metadata.py`, which already
detects generator-written provenance and handles it natively. Keeping the two
apart means there is exactly one implementation of the derivation, shared by
the legacy CSV and the regenerated runs.

Runs are read and written one at a time, so peak memory is one run rather than
the whole matrix.

Usage
-----
    # validate without writing anything
    python experiments/revision_2026/merge_runs.py --check-only

    # pool data/runs/*.csv into data/runs/gray-GOOSE-runs.parquet
    python experiments/revision_2026/merge_runs.py

    # then annotate, which adds event_id and split_group
    python experiments/revision_2026/add_experiment_metadata.py \
        --dataset data/runs/gray-GOOSE-runs.parquet
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_RUNS_DIR = os.path.join(REPO_ROOT, "data", "runs")
DEFAULT_BASENAME = "gray-GOOSE-runs"

# Written into every row by RunContext; kept in sync with NATIVE_COLUMNS in
# add_experiment_metadata.py. impairment_mode/impairment_rate/
# impairment_intensity_ms describe the card-C benign-degradation controls and
# are, like loss_rate/burst_size, constant within a run rather than per-row -
# a `normal` row inside a benign-impairment run still carries that run's mode.
NATIVE_COLUMNS = [
    "run_id", "trace_id", "batch_index", "scenario_id", "seed",
    "attack_variant", "loss_rate", "burst_size", "traffic_rate", "substation_config",
    "impairment_mode", "impairment_rate", "impairment_intensity_ms",
]

# Constant within a run, so a second distinct value means two runs got pooled
# into one file.
PER_RUN_CONSTANT = ["run_id", "trace_id", "scenario_id", "seed",
                    "attack_variant", "loss_rate", "burst_size",
                    "traffic_rate", "substation_config",
                    "impairment_mode", "impairment_rate", "impairment_intensity_ms"]


class RunError(Exception):
    """A run that must not be pooled."""


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover(runs_dir):
    """Find run CSVs, newest matrix manifest first if there is one."""
    if not os.path.isdir(runs_dir):
        raise SystemExit(f"Runs directory not found: {runs_dir}\n"
                         f"Generate runs first with generate_run_matrix.py.")

    paths = sorted(glob.glob(os.path.join(runs_dir, "*.csv")))
    if not paths:
        raise SystemExit(f"No .csv files in {runs_dir}.")
    return paths


def read_sidecar(csv_path):
    """The `.run.json` ERENO writes next to a dataset, or None."""
    path = csv_path + ".run.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def read_matrix_manifest(runs_dir):
    path = os.path.join(runs_dir, "run_matrix.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def payload_fingerprint(df):
    """Digest of everything except the provenance columns.

    Two runs are only independent units if their *messages* differ. Hashing the
    payload alone means a run duplicated under a new run_id and seed still
    collides, which is precisely the failure worth catching.
    """
    payload = df.drop(columns=[c for c in NATIVE_COLUMNS if c in df.columns])
    digest = hashlib.sha256()
    digest.update(",".join(payload.columns).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(payload, index=False).values.tobytes())
    return digest.hexdigest()


def validate(df, csv_path, sidecar):
    """Check one run in isolation. Raises RunError on anything disqualifying."""
    name = os.path.basename(csv_path)
    stem = os.path.splitext(name)[0]

    missing = [c for c in NATIVE_COLUMNS if c not in df.columns]
    if missing:
        raise RunError(
            f"{name}: missing provenance columns {missing}. This CSV predates the "
            f"RunContext patch - regenerate it."
        )

    if df.empty:
        raise RunError(f"{name}: no rows.")

    for column in PER_RUN_CONSTANT:
        values = df[column].unique()
        if len(values) != 1:
            raise RunError(
                f"{name}: `{column}` holds {len(values)} distinct values "
                f"({list(values)[:4]}...). A run file must describe exactly one run; "
                f"this one looks like two runs appended together."
            )

    run_id = str(df["run_id"].iloc[0])
    seed = int(df["seed"].iloc[0])

    if run_id != stem:
        raise RunError(
            f"{name}: rows say run_id={run_id!r} but the filename says {stem!r}. "
            f"The run was almost certainly generated with a stale params.properties."
        )

    if sidecar is not None:
        if str(sidecar.get("run_id")) != run_id:
            raise RunError(
                f"{name}: sidecar run_id={sidecar.get('run_id')!r} disagrees with the "
                f"rows ({run_id!r})."
            )
        if int(sidecar.get("seed", -1)) != seed:
            raise RunError(
                f"{name}: sidecar seed={sidecar.get('seed')} disagrees with the rows ({seed})."
            )

    return {
        "run_id": run_id,
        "seed": seed,
        "trace_id": str(df["trace_id"].iloc[0]),
        "scenario_id": str(df["scenario_id"].iloc[0]),
        "attack_variant": str(df["attack_variant"].iloc[0]),
        "loss_rate": float(df["loss_rate"].iloc[0]),
        "burst_size": int(df["burst_size"].iloc[0]),
        "impairment_mode": str(df["impairment_mode"].iloc[0]) if "impairment_mode" in df.columns else "NONE",
        "batches": int(df["batch_index"].max()) if "batch_index" in df.columns else None,
        "rows": len(df),
        # class-based, not attack_variant-based: attack_variant is "none" for
        # both `normal` and `benign_degradation` rows, so this counts any
        # labelled (non-normal) row - meaningful for attack and benign runs
        # alike, unlike a strict count of "attack" rows.
        "labelled_rows": int((df["class"] != "normal").sum()) if "class" in df.columns else None,
        "sidecar": sidecar is not None,
        "file": name,
    }


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------

def load_run(path):
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def merge(paths, out_path, fmt, check_only):
    """Validate every run, then stream them into one file."""
    writer = None
    schema = None
    header = None
    summaries = []
    fingerprints = {}
    problems = []

    for i, path in enumerate(paths, 1):
        name = os.path.basename(path)
        print(f"[{i}/{len(paths)}] {name}", flush=True)

        df = load_run(path)

        if header is None:
            header = list(df.columns)
        elif list(df.columns) != header:
            extra = set(df.columns) - set(header)
            gone = set(header) - set(df.columns)
            problems.append(
                f"{name}: schema differs from the first run "
                f"(extra={sorted(extra)}, missing={sorted(gone)}). Runs from different "
                f"ERENO builds cannot be pooled."
            )
            continue

        try:
            summary = validate(df, path, read_sidecar(path))
        except RunError as err:
            problems.append(str(err))
            continue

        fingerprint = payload_fingerprint(df)
        if fingerprint in fingerprints:
            problems.append(
                f"{name}: identical payload to {fingerprints[fingerprint]}. Two runs "
                f"carrying the same messages are one experimental unit, not two - check "
                f"that the seeds really reached the generator."
            )
            continue
        fingerprints[fingerprint] = name

        duplicate = next((s for s in summaries if s["run_id"] == summary["run_id"]), None)
        if duplicate is not None:
            problems.append(f"{name}: run_id {summary['run_id']!r} already pooled from "
                            f"{duplicate['file']}.")
            continue

        summary["fingerprint"] = fingerprint[:12]

        if not check_only:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            if fmt == "csv":
                first = not summaries
                df.to_csv(out_path, index=False, mode="w" if first else "a", header=first)
            else:
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(out_path, schema)
                else:
                    # A column that happens to be all-integer in one run and
                    # fractional in another arrives with a different Arrow type.
                    # Casting to the first run's schema keeps one file readable;
                    # a genuinely incompatible column is a generator bug, so it
                    # is reported rather than coerced.
                    try:
                        table = table.cast(schema)
                    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as err:
                        problems.append(
                            f"{name}: column types incompatible with the first run ({err}). "
                            f"Pool it separately or fix the generator."
                        )
                        continue
                writer.write_table(table)

        summaries.append(summary)
        print(f"    {summary['rows']:,} rows  seed={summary['seed']}  "
              f"{summary['attack_variant']}  loss={summary['loss_rate']:g}")

    if writer is not None:
        writer.close()

    return summaries, problems


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def build_report(runs_dir, out_path, summaries, problems, matrix, check_only):
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total = sum(s["rows"] for s in summaries)

    lines = [
        "# Run merge report - Gray-GOOSE",
        "",
        f"- Generated: {generated}",
        f"- Runs directory: `{runs_dir}`",
        f"- Output: {'(check only, nothing written)' if check_only else '`' + out_path + '`'}",
        f"- Runs pooled: **{len(summaries)}**",
        f"- Rows pooled: **{total:,}**",
        "",
    ]

    if problems:
        lines += [
            "## Rejected runs",
            "",
            "These were left out of the merge. A rejected run is a data-integrity",
            "problem, not a warning to skip past: pooling it would put a mislabelled",
            "or duplicated group into the cross-validation split.",
            "",
        ]
        lines += [f"- {p}" for p in problems]
        lines.append("")
    else:
        lines += ["> All runs passed validation.", ""]

    lines += [
        "## Runs",
        "",
        "| run_id | seed | variant | impairment_mode | loss_rate | burst_size | batches | rows | labelled rows | payload |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in sorted(summaries, key=lambda r: (r["attack_variant"], r["impairment_mode"], r["loss_rate"], r["seed"])):
        lines.append(
            f"| {s['run_id']} | {s['seed']} | {s['attack_variant']} | {s['impairment_mode']} | "
            f"{s['loss_rate']:g} | {s['burst_size']} | "
            f"{s['batches'] if s['batches'] is not None else '-'} | "
            f"{s['rows']:,} | "
            f"{format(s['labelled_rows'], ',') if s['labelled_rows'] is not None else '-'} | "
            f"`{s['fingerprint']}` |"
        )
    lines.append("")

    if summaries:
        frame = pd.DataFrame(summaries)
        # attack_variant is "NONE" for every benign-impairment run (it never
        # ran an attack), so grouping by attack_variant alone would collapse
        # all 7 benign mechanisms into one bucket. `family` picks whichever of
        # the two axes actually varies for that run.
        frame["family"] = frame["attack_variant"].where(
            frame["attack_variant"] != "NONE", "BENIGN:" + frame["impairment_mode"]
        )
        per_variant = frame.groupby("family").agg(
            runs=("run_id", "nunique"), seeds=("seed", "nunique"), rows=("rows", "sum")
        ).reset_index()

        lines += [
            "## Coverage per variant",
            "",
            "The variant each run was *configured* to perform, under ERENO's own enum",
            "names, or `BENIGN:<mode>` for a card-C benign-impairment run. The",
            "annotation step rewrites `attack_variant` to the paper's names and to a",
            "per-message reading, where both benign and normal rows are `none`.",
            "",
            "| variant | runs | distinct seeds | rows |",
            "|---|---:|---:|---:|",
        ]
        for _, r in per_variant.iterrows():
            lines.append(f"| {r['family']} | {r['runs']} | {r['seeds']} | {r['rows']:,} |")
        lines.append("")

        weakest = int(per_variant["runs"].min())
        if weakest >= 2:
            lines += [
                f"> Every variant spans at least {weakest} runs, so leaving one group out",
                "> still leaves that variant represented in training. This is the property",
                "> the submitted dataset lacked, and the reason grouped CV was impossible",
                "> on it.",
                "",
            ]
        else:
            lines += [
                "> **WARNING** at least one variant occupies a single run. Leaving that",
                "> group out removes the variant from training entirely, which is exactly",
                "> the defect of the submitted dataset. Generate more seeds per variant",
                "> before running grouped CV.",
                "",
            ]

        sidecars = int(frame["sidecar"].sum())
        if sidecars < len(frame):
            lines += [
                f"> {len(frame) - sidecars} run(s) have no `.run.json` sidecar, so their",
                "> provenance was checked against the rows and filename only.",
                "",
            ]

    if matrix is not None:
        failed = [r for r in matrix.get("runs", []) if r.get("status") not in ("ok", "skipped")]
        lines += [
            "## Matrix manifest",
            "",
            f"- Generated: {matrix.get('generated', 'unknown')}",
            f"- Cells in the matrix: {len(matrix.get('runs', []))}",
            f"- Cells that failed to generate: {len(failed)}",
            "",
        ]
        for r in failed:
            lines.append(f"- `{r.get('run_id')}`: {r.get('status')}")
        if failed:
            lines.append("")

    lines += [
        "## Next step",
        "",
        "`event_id` and `split_group` are not written here. Derive them with the",
        "annotation script, which detects the generator-written provenance and uses",
        "it natively:",
        "",
        "```bash",
        f"python experiments/revision_2026/add_experiment_metadata.py --dataset {out_path}",
        "```",
        "",
    ]
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR,
                   help="Directory holding one CSV per run (default: data/runs).")
    p.add_argument("--out", default=None,
                   help="Output path. Defaults to <runs-dir>/gray-GOOSE-runs.<format>.")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="Output format (default parquet).")
    p.add_argument("--report", default=None,
                   help="Path of the Markdown merge report (default: next to this script).")
    p.add_argument("--check-only", action="store_true",
                   help="Validate every run and report, without writing a dataset.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    runs_dir = os.path.abspath(args.runs_dir)
    out_path = args.out or os.path.join(runs_dir, f"{DEFAULT_BASENAME}.{args.format}")
    report_path = args.report or os.path.join(HERE, "merge_report.md")

    paths = discover(runs_dir)
    print(f"{len(paths)} run file(s) in {runs_dir}")

    summaries, problems = merge(paths, out_path, args.format, args.check_only)

    report = build_report(runs_dir, out_path, summaries, problems,
                          read_matrix_manifest(runs_dir), args.check_only)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report) + "\n")
    print(f"\nMerge report written: {report_path}")

    if problems:
        print(f"\n{len(problems)} run(s) REJECTED:")
        for p in problems:
            print(f"  - {p}")

    if not summaries:
        print("Nothing pooled.")
        return 1

    total = sum(s["rows"] for s in summaries)
    if args.check_only:
        print(f"--check-only: {len(summaries)} run(s), {total:,} rows would be pooled.")
    else:
        print(f"Wrote {out_path}: {len(summaries)} run(s), {total:,} rows.")
        print(f"\nNext: python experiments/revision_2026/add_experiment_metadata.py "
              f"--dataset {out_path}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
