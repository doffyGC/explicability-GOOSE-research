"""
Add experimental-unit metadata to the Gray-GOOSE dataset.

Checklist item A.3 ("Adicionar ao dataset: run_id, trace_id, event_id,
scenario_id, seed, attack_variant, loss_rate, burst_size, traffic_rate,
substation_config e split_group").

The submitted dataset (`data/CSV files/gray-GOOSE.csv`) is a flat, shuffled
message table with no provenance columns, so part of the requested metadata is
*derived* from the GOOSE protocol state carried in the rows and part can only
come from the ERENO generation run. This script does the first part, is
explicit about the second, and writes an audit report saying which is which.

How the derivation works
------------------------
In IEC 61850 GOOSE, StNum increments once per state change and t is the
timestamp of that state change. So inside a single publisher stream the pair
(StNum, t) is a bijection: it identifies one *event* - one state, with all its
retransmissions sharing the same t and increasing SqNum. That gives event_id
directly.

Grouping events back into traces is possible because each attack class in this
dataset turns out to be exactly one stream: within every attack class,
StNum -> t is strictly increasing with no repeats (checked and reported by
--audit-only). The rows labelled `normal` are a *pool* of the benign messages
of those same streams, so a normal row can be attributed to a trace when:

  1. its (StNum, t) pair is also carried by an attack row  -> direct anchor; or
  2. only one trace's known timeline can bracket it, since StNum -> t must stay
     strictly increasing inside a trace                    -> propagated anchor.

Anything that survives both tests is genuinely ambiguous from the CSV alone and
is labelled T-UNRESOLVED rather than guessed. Guessing would put the same real
stream on both sides of a grouped split and silently reintroduce the leakage
this revision is supposed to remove.

What CANNOT be derived
----------------------
scenario_id, seed, loss_rate, burst_size, traffic_rate and substation_config
are *inputs* to ERENO. They are not recoverable from the message table. Supply
them with --manifest manifest.json (see --write-manifest-template); without one
those columns are written as nulls and the audit report flags them.

Empirical estimates of loss / burst / traffic are printed in the audit report
for cross-checking but are deliberately NOT written into the dataset: a
measured rate is not the generation parameter and must not be mistaken for one.

Usage
-----
    python experiments/revision_2026/add_experiment_metadata.py --audit-only
    python experiments/revision_2026/add_experiment_metadata.py --write-manifest-template
    python experiments/revision_2026/add_experiment_metadata.py \
        --manifest experiments/revision_2026/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "data", "CSV files", "gray-GOOSE.csv")

NORMAL_LABEL = "normal"

# Raw ERENO class label -> short variant name used in the paper. The order
# matches CLASS_NAMES in config.py: LabelEncoder sorts labels alphabetically
# and lowercase "normal" sorts last.
BENIGN_DEGRADATION_LABEL = "benign_degradation"

VARIANT_OF_CLASS = {
    "DETERMINISTIC_BURST_ORIENTEDGRAYHOLE": "SAG.DB",
    "FULLY_RANDOMIZED_ORIENTEDGRAYHOLE": "FRG",
    "RANDOMIC_BURST_ORIENTEDGRAYHOLE": "SAG.PB",
    "RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE": "SAG.PBM",
    NORMAL_LABEL: "none",
    # benign_degradation (card C) is not an attack - attack_variant answers
    # "which attack is this message part of", so it reads "none" here exactly
    # like `normal`. The mechanism that produced it lives in `impairment_mode`
    # instead (see run_native): merging the two would make attack_variant lie
    # about a run that never fired an attack.
    BENIGN_DEGRADATION_LABEL: "none",
}

# Publisher identity. Constant in the submitted dataset, which is itself a
# finding to report: a single substation configuration.
SUBSTATION_KEY_COLUMNS = ["gocbRef", "datSet", "goID", "gooseAppid", "ethSrc", "ethDst"]

# ERENO's enum names -> the short names the paper uses. Applied in native mode
# so both modes speak the same vocabulary in the `attack_variant` column.
VARIANT_OF_ENUM = {
    "DETERMINISTIC_BURST": "SAG.DB",
    "FULLY_RANDOMIZED": "FRG",
    "RANDOMIC_BURST": "SAG.PB",
    "RANDOMIC_MESSAGE": "SAG.PBM",
}

UNRESOLVED = "T-UNRESOLVED"

# Added columns, in the order the checklist lists them.
METADATA_COLUMNS = [
    "run_id",
    "trace_id",
    "event_id",
    "scenario_id",
    "seed",
    "attack_variant",
    "loss_rate",
    "burst_size",
    "traffic_rate",
    "substation_config",
    "split_group",
]

# Per-scenario fields a manifest may provide. Every one of these is an ERENO
# *input*; none is inferable from the message table.
MANIFEST_FIELDS = ["scenario_id", "seed", "loss_rate", "burst_size", "traffic_rate"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_raw(path):
    """Load the dataset and normalise column names.

    The shipped CSV header has a leading-whitespace `    Time` column; every
    downstream step here assumes stripped names.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.endswith(".csv"):
        df = pd.read_csv(path, encoding="utf-8")
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        raise ValueError("File format not supported. Use .csv or .parquet.")

    df.columns = [c.strip() for c in df.columns]

    missing = {"StNum", "SqNum", "t", "Time", "class"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    unknown = set(df["class"].unique()) - set(VARIANT_OF_CLASS)
    if unknown:
        raise ValueError(
            f"Unknown class labels {sorted(unknown)}; extend VARIANT_OF_CLASS before running."
        )

    return df


# --------------------------------------------------------------------------
# Event / trace derivation
# --------------------------------------------------------------------------

def build_events(df):
    """Factorise rows into GOOSE events keyed by (StNum, t).

    Returns (row_event, events) where row_event holds one event index per
    dataset row and events is a DataFrame of the distinct (StNum, t) pairs.
    """
    codes, uniques = pd.MultiIndex.from_frame(df[["StNum", "t"]]).factorize()
    events = uniques.to_frame(index=False)
    events.columns = ["StNum", "t"]
    return np.asarray(codes, dtype=np.int64), events


def anchor_events(df, row_event, n_events, variants):
    """Label every event that carries attack rows with that attack's trace.

    An event carrying rows of two different attack classes cannot belong to a
    single stream, so it is left unanchored and counted as a conflict.
    """
    variant_index = {v: i for i, v in enumerate(variants)}
    is_attack = (df["class"] != NORMAL_LABEL).values

    row_variant = df["class"].map(VARIANT_OF_CLASS).map(variant_index).values

    pairs = pd.DataFrame(
        {"event": row_event[is_attack], "variant": row_variant[is_attack]}
    ).drop_duplicates()

    per_event = pairs.groupby("event")["variant"].agg(["first", "size"])

    anchor = np.full(n_events, -1, dtype=np.int64)
    clean = per_event[per_event["size"] == 1]
    anchor[clean.index.values] = clean["first"].values

    n_conflicts = int((per_event["size"] > 1).sum())
    return anchor, n_conflicts


def attribute_traces(events, anchor, n_traces, max_iterations=20):
    """Propagate trace labels from anchored to unanchored events.

    Inside one trace StNum -> t is strictly increasing, so an unanchored event
    (s, t0) is *feasible* for trace j only if trace j's known timeline brackets
    it: the largest known StNum below s must have t <= t0, the smallest known
    StNum above s must have t >= t0, and if s is already known in trace j its t
    must match exactly. Events feasible for exactly one trace are assigned; the
    rest stay unresolved.

    Assignments only tighten the brackets, so the pass is repeated until it
    stops gaining events.
    """
    st = events["StNum"].values.astype(np.int64)
    tt = events["t"].values.astype(np.float64)
    label = anchor.copy()

    history = []
    for iteration in range(max_iterations):
        feasible = np.zeros((len(events), n_traces), dtype=bool)

        for j in range(n_traces):
            member = label == j
            s_j = st[member]
            t_j = tt[member]
            if len(s_j) == 0:
                continue

            order = np.argsort(s_j, kind="stable")
            s_j, t_j = s_j[order], t_j[order]

            lo = np.searchsorted(s_j, st, side="left")
            hi = np.searchsorted(s_j, st, side="right")

            lower = np.where(lo > 0, t_j[np.clip(lo - 1, 0, len(t_j) - 1)], -np.inf)
            upper = np.where(hi < len(t_j), t_j[np.clip(hi, 0, len(t_j) - 1)], np.inf)
            ok = (tt >= lower) & (tt <= upper)

            # StNum already known in this trace: t has to match it exactly.
            exact = lo < hi
            ok[exact] = np.isclose(t_j[np.clip(lo[exact], 0, len(t_j) - 1)], tt[exact])

            feasible[:, j] = ok

        n_feasible = feasible.sum(axis=1)
        newly = (label < 0) & (n_feasible == 1)
        gained = int(newly.sum())
        if gained:
            label[newly] = feasible[newly].argmax(axis=1)

        history.append({"iteration": iteration, "gained": gained,
                        "assigned": int((label >= 0).sum())})
        if gained == 0:
            break

    stats = {
        "iterations": history,
        "n_events": int(len(events)),
        "n_anchored": int((anchor >= 0).sum()),
        "n_assigned": int((label >= 0).sum()),
        "n_unresolved": int((label < 0).sum()),
    }
    return label, stats


# A stream may carry a few duplicated states (the generator occasionally
# re-publishes one 0.1 s later). Below this fraction they are treated as noise;
# above it, the class is genuinely more than one stream and the trace
# attribution cannot be trusted.
MULTI_T_TOLERANCE = 1e-3


def build_event_ids(events, label, trace_names, row_event, index):
    """Build one stable, human-readable id per event, then map it onto rows.

    The natural form is `<trace_id>-E<StNum>`: readable, and it ties a message
    straight back to the GOOSE state it belongs to. It is unique wherever
    (StNum, t) is a bijection, which is the resolved traces. Two cases break
    that and get the event index appended:

      * the unresolved pool, which mixes states from several traces that can
        share a StNum;
      * the handful of states the generator re-published ~0.1 s later, which
        appear as one StNum with two distinct t.

    Disambiguating only the colliding ids keeps the rest readable, and the
    assertion below makes the guarantee explicit rather than assumed.
    """
    trace_per_event = np.array(
        [trace_names[i] if i >= 0 else UNRESOLVED for i in label], dtype=object
    )
    ids = pd.Series(trace_per_event) + "-E" + events["StNum"].astype(str).values

    duplicated = ids.duplicated(keep=False).values
    if duplicated.any():
        codes = pd.Series(np.arange(len(events)).astype(str))
        ids = ids.mask(duplicated, ids + "-" + codes)

    assert ids.is_unique, "event_id is not unique after disambiguation"
    return pd.Series(ids.values[row_event], index=index, dtype="object")


def check_single_stream_per_variant(df, variants):
    """Verify the assumption the whole derivation rests on: each attack class
    is one stream, i.e. StNum -> t is a strictly increasing bijection within it.
    """
    report = {}
    for variant in variants:
        raw = [k for k, v in VARIANT_OF_CLASS.items() if v == variant][0]
        g = df[df["class"] == raw]
        per_stnum = g.groupby("StNum")["t"].nunique()
        ordered = g.groupby("StNum")["t"].first().sort_index()
        n_multi = int((per_stnum > 1).sum())
        frac_multi = n_multi / len(per_stnum) if len(per_stnum) else 0.0
        monotonic = bool(ordered.is_monotonic_increasing)
        report[variant] = {
            "rows": int(len(g)),
            "distinct_stnum": int(len(per_stnum)),
            "stnum_with_multiple_t": n_multi,
            "frac_stnum_with_multiple_t": frac_multi,
            "max_t_per_stnum": int(per_stnum.max()),
            "t_monotonic_in_stnum": monotonic,
            "single_stream": bool(monotonic and frac_multi < MULTI_T_TOLERANCE),
        }
    return report


# --------------------------------------------------------------------------
# Substation configuration
# --------------------------------------------------------------------------

def derive_substation_config(df, override=None):
    """Derive a substation-configuration label from the publisher identity."""
    present = [c for c in SUBSTATION_KEY_COLUMNS if c in df.columns]
    combos = df[present].drop_duplicates() if present else pd.DataFrame()

    if override:
        labels = pd.Series(override, index=df.index, dtype="object")
        return labels, present, combos, override

    if len(combos) <= 1:
        # One publisher for the whole dataset. Flat label, and the audit report
        # calls this out: the dataset has no traffic/topology diversity.
        return pd.Series("SUB-A", index=df.index, dtype="object"), present, combos, "SUB-A"

    keys = df[present].astype(str).agg("|".join, axis=1)
    codes, _ = pd.factorize(keys)
    labels = pd.Series([f"SUB-{chr(ord('A') + c)}" for c in codes], index=df.index, dtype="object")
    return labels, present, combos, None


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def manifest_template(variants, trace_ids):
    """Build a manifest skeleton for the user to fill in from ERENO."""
    return {
        "_README": (
            "Generation parameters for the Gray-GOOSE dataset. These come from the "
            "ERENO run configuration, NOT from the message table. Fill every null "
            "from the generator config/scripts, or regenerate the dataset so the "
            "values are emitted directly. Keys under 'scenarios' are the scenario a trace "
            "was generated under; keys under 'traces' override per trace_id."
        ),
        "substation_config": None,
        "scenarios": {
            v: {f: None for f in MANIFEST_FIELDS} for v in variants + ["UNRESOLVED"]
        },
        "traces": {t: {f: None for f in MANIFEST_FIELDS} for t in trace_ids},
    }


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object.")
    return manifest


def resolve_scenario_fields(manifest, scenario_key, trace_series):
    """Resolve the manifest-supplied columns per row.

    `scenario_key` is the *trace's* scenario, not the row's label: a benign
    message captured inside the SAG.DB run was still generated under the SAG.DB
    scenario, with that run's seed and loss rate. Keying on the row's `class`
    instead would hand every benign row the same null scenario and lose the
    link between a message and the run that produced it.

    Per-trace manifest entries take precedence over per-scenario entries;
    anything absent stays null.
    """
    scenarios = (manifest or {}).get("scenarios", {}) or {}
    traces = (manifest or {}).get("traces", {}) or {}

    resolved = {}
    for field in MANIFEST_FIELDS:
        by_scenario = {k: v.get(field) for k, v in scenarios.items() if isinstance(v, dict)}
        by_trace = {k: v.get(field) for k, v in traces.items() if isinstance(v, dict)}

        values = scenario_key.map({k: v for k, v in by_scenario.items() if v is not None})
        if by_trace:
            override = trace_series.map({k: v for k, v in by_trace.items() if v is not None})
            values = override.where(override.notna(), values)
        resolved[field] = values

    # scenario_id has a usable fallback: one scenario per trace.
    fallback = "SC-" + scenario_key.astype(str)
    resolved["scenario_id"] = resolved["scenario_id"].where(
        resolved["scenario_id"].notna(), fallback
    )
    return resolved


# --------------------------------------------------------------------------
# Diagnostics (report only - never written into the dataset)
# --------------------------------------------------------------------------

def trace_diagnostics(df, trace_ids, variants):
    """Empirical per-trace measurements, for cross-checking a manifest.

    These are observations of the delivered stream, not ERENO parameters.
    """
    rows = []
    for trace in sorted(set(trace_ids)):
        mask = trace_ids == trace
        g = df.loc[mask]
        span = float(g["Time"].max() - g["Time"].min()) if len(g) else 0.0
        states = g.groupby("StNum")["t"].first().sort_index()
        gaps = np.diff(states.index.values) if len(states) > 1 else np.array([])
        dropped = gaps[gaps > 1] - 1 if len(gaps) else np.array([])

        rows.append(
            {
                "trace_id": trace,
                "messages": int(len(g)),
                "events": int(len(states)),
                "time_span_s": round(span, 1),
                "msgs_per_s": round(len(g) / span, 4) if span > 0 else float("nan"),
                "state_gap_fraction": round(float((gaps > 1).mean()), 4) if len(gaps) else float("nan"),
                "dropped_states_mean": round(float(dropped.mean()), 3) if len(dropped) else 0.0,
                "dropped_states_max": int(dropped.max()) if len(dropped) else 0,
                "variants": ", ".join(sorted(set(df.loc[mask, "class"].map(VARIANT_OF_CLASS)))),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(path, sections):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(sections) + "\n")


def build_report(args, df, stream_check, anchor_stats, n_conflicts, trace_ids,
                 substation_info, manifest, resolved, diagnostics, out_path):
    present, combos, flat_label = substation_info
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("# Dataset metadata audit - Gray-GOOSE")
    lines.append("")
    lines.append(f"- Generated: {generated}")
    lines.append(f"- Source dataset: `{os.path.relpath(args.dataset, REPO_ROOT)}`")
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Output: {'(audit only, nothing written)' if args.audit_only else '`' + os.path.relpath(out_path, REPO_ROOT) + '`'}")
    lines.append(f"- Manifest: {'`' + os.path.relpath(args.manifest, REPO_ROOT) + '`' if args.manifest else 'NONE - generation parameters left null'}")
    lines.append(f"- split_group level: `{args.split_level}`")
    lines.append("")

    lines.append("## 1. Assumption check: is each attack class a single stream?")
    lines.append("")
    lines.append("The trace derivation is only valid if, within an attack class, `StNum -> t`")
    lines.append("is a strictly increasing bijection (one publisher stream).")
    lines.append("")
    lines.append("| attack_variant | rows | distinct StNum | StNum with >1 t | share | t monotonic in StNum | single stream |")
    lines.append("|---|---:|---:|---:|---:|:--:|:--:|")
    for variant, info in stream_check.items():
        lines.append(
            f"| {variant} | {info['rows']:,} | {info['distinct_stnum']:,} | "
            f"{info['stnum_with_multiple_t']:,} | {info['frac_stnum_with_multiple_t']:.4%} | "
            f"{'yes' if info['t_monotonic_in_stnum'] else 'NO'} | "
            f"{'yes' if info['single_stream'] else 'NO'} |"
        )
    lines.append("")
    lines.append(f"A few duplicated states (below {MULTI_T_TOLERANCE:.1%} of a class) are treated as")
    lines.append("generator noise: the same StNum re-published ~0.1 s later with overlapping SqNum.")
    lines.append("They are a data-quality point worth a line in the data card, not a second stream.")
    lines.append("")
    if not all(i["single_stream"] for i in stream_check.values()):
        lines.append("> **WARNING** at least one attack class is not a single stream. The")
        lines.append("> trace attribution below is unsafe - do not use it for grouped splits.")
        lines.append("")

    lines.append("## 2. Event and trace attribution")
    lines.append("")
    lines.append(f"- Distinct GOOSE events, keyed `(StNum, t)`: **{anchor_stats['n_events']:,}**")
    lines.append(f"- Events anchored directly by an attack row: **{anchor_stats['n_anchored']:,}**")
    lines.append(f"- Events resolved after bracket propagation: **{anchor_stats['n_assigned']:,}** "
                 f"({anchor_stats['n_assigned'] / anchor_stats['n_events']:.1%})")
    lines.append(f"- Events left ambiguous (`{UNRESOLVED}`): **{anchor_stats['n_unresolved']:,}** "
                 f"({anchor_stats['n_unresolved'] / anchor_stats['n_events']:.1%})")
    lines.append(f"- Events carrying two different attack classes (conflicts): **{n_conflicts:,}**")
    lines.append("")
    lines.append("Propagation passes:")
    lines.append("")
    lines.append("| pass | events gained | events assigned |")
    lines.append("|---:|---:|---:|")
    for it in anchor_stats["iterations"]:
        lines.append(f"| {it['iteration']} | {it['gained']:,} | {it['assigned']:,} |")
    lines.append("")

    counts = pd.Series(trace_ids).value_counts().sort_index()
    lines.append("Rows per trace:")
    lines.append("")
    lines.append("| trace_id | rows | share |")
    lines.append("|---|---:|---:|")
    for trace, n in counts.items():
        lines.append(f"| {trace} | {n:,} | {n / len(df):.1%} |")
    lines.append("")

    n_groups = int(counts.index.drop(UNRESOLVED, errors="ignore").nunique())
    unresolved_rows = int(counts.get(UNRESOLVED, 0))
    lines.append(f"> **Independent experimental units available: {n_groups}.**")
    lines.append(f"> `{UNRESOLVED}` is not a sixth unit - it is a pool of messages from all")
    lines.append("> traces that the CSV cannot separate, so it cannot go on one side of a split.")
    lines.append(f"> That leaves {n_groups} usable groups covering "
                 f"{(len(df) - unresolved_rows) / len(df):.1%} of the rows, one per attack scenario.")
    lines.append("> A GroupKFold over this many groups gives very few, highly correlated folds,")
    lines.append("> and each group is tied to exactly one attack class, so a left-out group is a")
    lines.append("> left-out *class* - the fold has no positive examples of what it is tested on.")
    lines.append("> Regenerating the dataset with several ERENO runs per scenario (different")
    lines.append("> seeds, durations and traffic profiles) is the only way to get the number of")
    lines.append("> independent units the grouped-statistics part of the revision needs.")
    lines.append("")

    lines.append("## 3. Substation configuration")
    lines.append("")
    lines.append(f"- Identity columns inspected: {', '.join('`' + c + '`' for c in present) if present else 'none present'}")
    lines.append(f"- Distinct publisher configurations found: **{len(combos)}**")
    if flat_label is not None and len(combos) <= 1:
        lines.append("")
        lines.append(f"> The whole dataset uses one publisher (`{flat_label}`): a single GOOSE control")
        lines.append("> block, dataset, appID and MAC pair. `substation_config` is therefore constant")
        lines.append("> and carries no information. This is the 'traffic diversity' gap the reviewers")
        lines.append("> raised, and it cannot be fixed by annotation - only by regeneration.")
    lines.append("")

    lines.append("## 4. Generation parameters (manifest-supplied)")
    lines.append("")
    lines.append("`seed`, `loss_rate`, `burst_size` and `traffic_rate` are ERENO *inputs*. They")
    lines.append("cannot be recovered from a message table and are only written when a manifest")
    lines.append("supplies them.")
    lines.append("")
    lines.append("| column | non-null rows | status |")
    lines.append("|---|---:|---|")
    for field in MANIFEST_FIELDS:
        filled = int(resolved[field].notna().sum())
        status = "supplied" if filled else "**MISSING - fill the manifest or regenerate**"
        if field == "scenario_id" and filled == len(df):
            status = "supplied" if manifest else "derived fallback (`SC-<trace scenario>`)"
        lines.append(f"| {field} | {filled:,} | {status} |")
    lines.append("")

    lines.append("## 5. Empirical diagnostics (NOT written into the dataset)")
    lines.append("")
    lines.append("Measurements of the delivered stream, useful for sanity-checking a manifest.")
    lines.append("A measured rate is not the generation parameter - do not copy these into")
    lines.append("`loss_rate` / `burst_size` / `traffic_rate`. The `" + UNRESOLVED + "` row mixes")
    lines.append("several traces, so its figures describe the pool, not any one run.")
    lines.append("")
    lines.append("| trace_id | messages | events | span (s) | msgs/s | states with a gap | mean states dropped | max states dropped | variants |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, r in diagnostics.iterrows():
        lines.append(
            f"| {r['trace_id']} | {r['messages']:,} | {r['events']:,} | {r['time_span_s']:,} | "
            f"{r['msgs_per_s']} | {r['state_gap_fraction']} | {r['dropped_states_mean']} | "
            f"{r['dropped_states_max']} | {r['variants']} |"
        )
    lines.append("")

    lines.append("## 6. What still blocks checklist item A.3")
    lines.append("")
    lines.append("- [ ] `seed`, `loss_rate`, `burst_size`, `traffic_rate` per scenario (ERENO config).")
    lines.append("- [ ] `substation_config` diversity: the dataset has a single publisher.")
    lines.append(f"- [ ] {anchor_stats['n_unresolved']:,} events cannot be attributed to a trace from the CSV alone.")
    lines.append("- [ ] Enough independent runs for grouped CV and per-group confidence intervals.")
    lines.append("")
    lines.append("All four are properties of the generation run. Annotating the delivered CSV")
    lines.append("cannot create them; regenerating with ERENO while emitting the identifiers")
    lines.append("can, and is the intent of Phase 1 of the revision plan.")

    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Native mode: the generator already emitted provenance
# --------------------------------------------------------------------------

# Columns the patched ERENO writes into every row (RunContext.csvHeader).
# impairment_mode/impairment_rate/impairment_intensity_ms describe the card-C
# benign-degradation controls; they are run-level constants exactly like
# loss_rate/burst_size (see BENIGN_DEGRADATION_LABEL handling below).
NATIVE_COLUMNS = [
    "run_id", "trace_id", "batch_index", "scenario_id", "seed",
    "attack_variant", "loss_rate", "burst_size", "traffic_rate", "substation_config",
    "impairment_mode", "impairment_rate", "impairment_intensity_ms",
]

# Enough of them to trust the row's own provenance over anything we could infer.
NATIVE_REQUIRED = ["run_id", "trace_id", "scenario_id", "seed", "attack_variant"]


def has_native_metadata(df):
    return all(c in df.columns for c in NATIVE_REQUIRED)


def build_event_ids_native(df):
    """Derive `event_id` from provenance the generator already wrote.

    A trace is one publisher stream, so `(trace_id, StNum, t)` identifies one
    state and all its retransmissions. Only genuinely colliding ids get the
    event index appended, which keeps the rest readable.
    """
    codes, uniques = pd.MultiIndex.from_frame(df[["trace_id", "StNum", "t"]]).factorize()
    events = uniques.to_frame(index=False)
    events.columns = ["trace_id", "StNum", "t"]

    ids = events["trace_id"].astype(str) + "-E" + events["StNum"].astype(str)
    duplicated = ids.duplicated(keep=False).values
    if duplicated.any():
        ids = ids.mask(duplicated, ids + "-" + pd.Series(np.arange(len(events)).astype(str)))

    assert ids.is_unique, "event_id is not unique after disambiguation"
    return pd.Series(ids.values[codes], index=df.index, dtype="object"), len(events)


def run_native(args, df, out_path, report_path):
    """Annotate a dataset that already carries generator provenance."""
    print("Native provenance detected - using the generator's own identifiers.")

    # The generator writes the *run's* variant into every row, benign rows
    # included, so its column answers "which attack was this run configured to
    # perform". The legacy mode's column answers "what is this message", with
    # `none` on benign rows. Same name, two meanings, and a pipeline that used
    # `attack_variant` as a label or a stratification key would behave
    # differently on the two datasets.
    #
    # Row-level wins, because it is the reading that survives pooling: once runs
    # of different variants sit in one table, a benign row carrying an attack
    # name would claim to be something it is not. Nothing is lost - the run's
    # configured variant stays in `scenario_id` (SC-<VARIANT>-l<loss>-b<burst>)
    # and is reported per run below.
    raw_variants = sorted(df["attack_variant"].astype(str).unique())
    df = df.copy()
    run_variant = (
        df.groupby("run_id")["attack_variant"]
        .first()
        .map(lambda v: VARIANT_OF_ENUM.get(str(v), str(v)))
    )
    df["attack_variant"] = df["class"].map(VARIANT_OF_CLASS)

    event_id, n_events = build_event_ids_native(df)
    print(f"  {n_events:,} distinct events across {df['run_id'].nunique():,} runs")

    split_source = {
        "run": df["run_id"],
        "trace": df["trace_id"],
        "event": event_id,
    }[args.split_level]

    meta = pd.DataFrame({"event_id": event_id, "split_group": split_source}, index=df.index)

    # Keep the checklist's column order, with the generator's values passed
    # through untouched and only the two derived columns added.
    ordered = [c for c in METADATA_COLUMNS if c in df.columns or c in meta.columns]
    out = pd.concat(
        [pd.concat([df[[c for c in ordered if c in df.columns]], meta], axis=1)[ordered],
         df.drop(columns=[c for c in ordered if c in df.columns])],
        axis=1,
    )

    has_impairment = "impairment_mode" in df.columns
    runs_agg = dict(
        rows=("run_id", "size"),
        scenario=("scenario_id", "first"),
        seed=("seed", "first"),
        loss_rate=("loss_rate", "first"),
        burst_size=("burst_size", "first"),
    )
    if has_impairment:
        runs_agg["impairment_mode"] = ("impairment_mode", "first")
    runs = df.groupby("run_id").agg(**runs_agg).reset_index()
    # From the generator's own column, not from the rows: `attack_variant` is
    # row-level now, and most rows in any run are benign. A benign-impairment
    # run's raw attack_variant is the constant "NONE" (it never ran an
    # attack), which would make every benign run indistinguishable in this
    # table - impairment_mode carries the mechanism instead, so prefer it.
    runs["variant"] = runs["run_id"].map(run_variant)
    if has_impairment:
        benign = runs["impairment_mode"].notna() & (runs["impairment_mode"] != "NONE")
        runs.loc[benign, "variant"] = "BENIGN:" + runs.loc[benign, "impairment_mode"]
    # class-based, not attack_variant-based: attack_variant is "none" for both
    # `normal` and `benign_degradation` rows (neither is an attack), so this
    # counts any labelled (non-normal) row - the same quantity that balances
    # both attack and benign-impairment runs (countMaliciousMessages on the
    # generator side does the equivalent thing for its batch target).
    runs["labelled_rows"] = runs["run_id"].map(
        df[df["class"] != NORMAL_LABEL].groupby("run_id").size()
    ).fillna(0).astype(int)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Dataset metadata audit - Gray-GOOSE (native mode)",
        "",
        f"- Generated: {generated}",
        f"- Source dataset: `{args.dataset}`",
        f"- Rows: {len(df):,}",
        f"- Output: `{out_path}`",
        f"- split_group level: `{args.split_level}`",
        "",
        "## 1. Provenance",
        "",
        "Every checklist column except `event_id` and `split_group` was written by",
        "the generator itself, so nothing here is inferred. `event_id` comes from",
        "`(trace_id, StNum, t)` and `split_group` mirrors the chosen level.",
        "",
        f"- Independent runs: **{df['run_id'].nunique():,}**",
        f"- Traces: **{df['trace_id'].nunique():,}**",
        f"- Scenarios: **{df['scenario_id'].nunique():,}**",
        f"- Distinct seeds: **{df['seed'].nunique():,}**",
        f"- Substation configurations: **{df['substation_config'].nunique():,}**",
        f"- Distinct events: **{n_events:,}**",
        f"- Variant names as emitted: {', '.join('`' + v + '`' for v in raw_variants)}",
        "",
        "> `attack_variant` describes the **message**, so benign rows carry `none`,",
        "> exactly as in legacy mode - the two datasets can be pooled and compared",
        "> column for column. The generator instead writes the *run's* configured",
        "> variant into every row; that reading is preserved in `scenario_id` and in",
        "> the `variant` column of the table below.",
        "",
        "## 2. Runs",
        "",
        "| run_id | scenario_id | seed | variant | loss_rate | burst_size | rows | labelled rows |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in runs.iterrows():
        lines.append(
            f"| {r['run_id']} | {r['scenario']} | {r['seed']} | {r['variant']} | "
            f"{r['loss_rate']} | {r['burst_size']} | {r['rows']:,} | {r['labelled_rows']:,} |"
        )
    lines += [
        "",
        "## 3. Class distribution per run",
        "",
        "| run_id | " + " | ".join(sorted(df["class"].unique())) + " |",
        "|---" * (1 + df["class"].nunique()) + "|",
    ]
    ct = pd.crosstab(df["run_id"], df["class"])
    for run_id, row in ct.iterrows():
        lines.append(f"| {run_id} | " + " | ".join(f"{row.get(c, 0):,}" for c in sorted(df["class"].unique())) + " |")

    lines += [
        "",
        "## 4. Grouped validation readiness",
        "",
        f"- `split_group` has **{split_source.nunique():,}** distinct values.",
    ]
    # Benign rows are in every run by construction, so counting `none` here
    # would report a coverage the attack classes do not have.
    per_variant = df[df["attack_variant"] != "none"].groupby("attack_variant")["run_id"].nunique()
    if not len(per_variant):
        # A benign-only dataset (card C) has zero attack rows by design - not
        # a coverage gap, so this must not print the single-run warning below.
        lines.append("- No attack-variant rows present in this dataset (benign-only pool).")
        lines.append("")
    else:
        lines += [
            f"- Each attack variant appears in **{per_variant.min():,}**"
            " run(s) at minimum, counting only runs that actually contain that attack.",
            "",
        ]
        if per_variant.min() >= 2:
            lines.append("> Every attack variant spans at least two runs, so a left-out group still")
            lines.append("> leaves that variant represented in training. GroupKFold is usable.")
        else:
            lines.append("> **WARNING** at least one attack variant occupies a single run, so leaving")
            lines.append("> that group out removes the variant from training entirely. Generate more")
        lines.append("> seeds per variant before running grouped CV.")
    lines.append("")

    if has_impairment and (df["impairment_mode"] != "NONE").any():
        per_mode = (
            df[df["impairment_mode"] != "NONE"]
            .groupby("impairment_mode")["run_id"]
            .nunique()
        )
        lines += [
            "## 5. Benign-degradation (card C) coverage",
            "",
            "`impairment_mode` is run-level (every row of a benign-impairment run",
            "carries it, `normal` rows included), so it is the axis to check here -",
            "`class` alone cannot distinguish which mechanism produced a run.",
            "",
            f"- Each impairment mode appears in **{per_mode.min():,}** run(s) at minimum.",
            "",
        ]
        if per_mode.min() >= 2:
            lines.append("> Every impairment mode spans at least two runs, so it can be left out of a")
            lines.append("> fold independently, same as an attack variant.")
        else:
            lines.append("> **WARNING** at least one impairment mode occupies a single run. Generate")
            lines.append("> more seeds for it before relying on grouped CV or LOETO over this axis.")
        lines.append("")

    write_report(report_path, lines)
    print(f"Audit report written: {report_path}")

    if args.audit_only:
        print("--audit-only: no dataset written.")
        return 0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Writing {out_path} ...")
    if args.format == "parquet":
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"  {len(out):,} rows x {out.shape[1]} columns")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Source dataset (.csv or .parquet).")
    p.add_argument("--out", default=None, help="Output path. Defaults next to the source with a -metadata suffix.")
    p.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                   help="Output format (default parquet: the annotated table is much larger as CSV).")
    p.add_argument("--manifest", default=None, help="JSON file with the ERENO generation parameters.")
    p.add_argument("--write-manifest-template", action="store_true",
                   help="Write a manifest skeleton next to this script and exit.")
    p.add_argument("--split-level", choices=["run", "trace", "event"], default="run",
                   help="Which identifier split_group mirrors (default: run).")
    p.add_argument("--report", default=None, help="Path of the Markdown audit report.")
    p.add_argument("--audit-only", action="store_true", help="Analyse and report without writing a dataset.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print(f"Loading {args.dataset} ...")
    df = load_raw(args.dataset)
    print(f"  {len(df):,} rows x {df.shape[1]} columns")

    variants = [VARIANT_OF_CLASS[c] for c in sorted(VARIANT_OF_CLASS) if c != NORMAL_LABEL]
    trace_names = {i: f"T{i:02d}-{v}" for i, v in enumerate(variants)}

    base, _ = os.path.splitext(args.dataset)
    out_path = args.out or f"{base}-metadata.{args.format}"
    report_path = args.report or os.path.join(HERE, "metadata_audit.md")

    # A dataset produced by the patched ERENO carries its own provenance, so
    # there is nothing to reverse-engineer and nothing to take on faith from a
    # manifest. Reconstruction is only for the legacy CSV.
    if has_native_metadata(df):
        return run_native(args, df, out_path, report_path)

    if args.write_manifest_template:
        path = os.path.join(HERE, "manifest.template.json")
        template = manifest_template(variants, list(trace_names.values()) + [UNRESOLVED])
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(template, fh, indent=2)
        print(f"Manifest template written: {path}")
        print("Fill it from the ERENO run configuration, then re-run with --manifest.")
        return 0

    print("Checking the single-stream assumption per attack class ...")
    stream_check = check_single_stream_per_variant(df, variants)
    for variant, info in stream_check.items():
        flag = "ok" if info["single_stream"] else "NOT A SINGLE STREAM"
        print(f"  {variant:<9} rows={info['rows']:>7,}  {flag}")

    print("Deriving events from (StNum, t) ...")
    row_event, events = build_events(df)
    print(f"  {len(events):,} distinct events")

    print("Anchoring events to traces ...")
    anchor, n_conflicts = anchor_events(df, row_event, len(events), variants)
    print(f"  {int((anchor >= 0).sum()):,} anchored, {n_conflicts:,} conflicting")

    print("Propagating trace labels ...")
    label, anchor_stats = attribute_traces(events, anchor, len(variants))
    print(f"  {anchor_stats['n_assigned']:,}/{anchor_stats['n_events']:,} events resolved "
          f"({anchor_stats['n_unresolved']:,} left as {UNRESOLVED})")

    # Per-row metadata.
    event_label = label[row_event]
    trace_ids = pd.Series(
        [trace_names[i] if i >= 0 else UNRESOLVED for i in event_label], index=df.index, dtype="object"
    )
    attack_variant = df["class"].map(VARIANT_OF_CLASS)

    # One ERENO run per trace in this dataset; run_id mirrors trace_id and the
    # audit report records the assumption.
    run_ids = trace_ids.copy()

    event_id = build_event_ids(events, label, trace_names, row_event, df.index)

    # The scenario follows the run, not the row's label: benign messages
    # captured inside an attack run belong to that run's scenario.
    scenario_key = pd.Series(
        [variants[i] if i >= 0 else "UNRESOLVED" for i in event_label],
        index=df.index,
        dtype="object",
    )

    manifest = load_manifest(args.manifest) if args.manifest else None
    resolved = resolve_scenario_fields(manifest, scenario_key, trace_ids)

    substation, present, combos, flat_label = derive_substation_config(
        df, (manifest or {}).get("substation_config")
    )

    split_source = {"run": run_ids, "trace": trace_ids, "event": event_id}[args.split_level]

    meta = pd.DataFrame(
        {
            "run_id": run_ids,
            "trace_id": trace_ids,
            "event_id": event_id,
            "scenario_id": resolved["scenario_id"],
            "seed": resolved["seed"],
            "attack_variant": attack_variant,
            "loss_rate": resolved["loss_rate"],
            "burst_size": resolved["burst_size"],
            "traffic_rate": resolved["traffic_rate"],
            "substation_config": substation,
            "split_group": split_source,
        },
        index=df.index,
    )[METADATA_COLUMNS]

    diagnostics = trace_diagnostics(df, trace_ids, variants)

    report = build_report(args, df, stream_check, anchor_stats, n_conflicts, trace_ids,
                          (present, combos, flat_label), manifest, resolved, diagnostics, out_path)
    write_report(report_path, report)
    print(f"Audit report written: {report_path}")

    if args.audit_only:
        print("--audit-only: no dataset written.")
        return 0

    out = pd.concat([meta, df], axis=1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Writing {out_path} ...")
    if args.format == "parquet":
        out.to_parquet(out_path, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"  {len(out):,} rows x {out.shape[1]} columns")

    if not args.manifest:
        print()
        print("NOTE: no manifest supplied, so seed / loss_rate / burst_size / traffic_rate")
        print("      are null. Run --write-manifest-template, fill it from the ERENO")
        print("      configuration, and re-run with --manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
