"""What the lost GOOSE frames do to a transfer trip. Card G, items 3-4.

Checklist ref.: G.3 (connect GOOSE loss to a protection function) at the level
G.4 allows - a simple model as preliminary evidence, not proof. The design,
the constants and the expected readings are preregistered in
`protection_consequence.md` §§1-6, written before this script ran.

The function
------------
The scenario's GOOSE is read as a **direct transfer trip**: the fault state
(`cbStatus = 1`) is the trip command, and the remote relay trips on the first
delivered frame that carries it. The unit is one fault event - a fault state
and the recovery the generator schedules 100 ms later.

The dataset only holds delivered frames, so the events an attack hid are
reconstructed from the `StNum` sequence (`fault_events`): per trace, every
`StNum` of fault parity strictly between the first and the last state, with
the event time taken from the fault state's own `t` field or, when no frame
of it survived, from the recovery's `t` minus 0.1 s. An event with neither is
*unobservable* - counted, and excluded from every rate.

Two layers
----------
1. **Subscriber.** Missed trip (no fault frame delivered), transfer latency
   `L` (first delivered fault frame minus event time), and the rate at which
   `L` breaks the 3 ms and 10 ms transfer-time requirements for trip messages.
   A missed trip breaks both.
2. **Power system.** Clearing time `T_relay + L + T_breaker` when the trip
   arrives, `T_backup + T_breaker` when it does not, and relative fault energy
   `T_clear / (T_relay + T_breaker)` - the `I²t` a sustained fault delivers
   relative to an ideally delivered trip, which is also the voltage-sag
   duration ratio. Layer 2 is a deterministic transform of layer 1 through
   three assumed constants: it adds units, not information.

Usage
-----
    python experiments/revision_2026/protection_consequence.py \\
      --dataset data/runs/gray-GOOSE-runs-prepared.parquet \\
      --out experiments/revision_2026/protection_consequence_result.md \\
      --json-out experiments/revision_2026/protection_consequence_result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

COLUMNS = ["run_id", "trace_id", "burst_size", "loss_rate",
           "StNum", "cbStatus", "t", "GooseTimestamp"]

# The generator schedules the recovery exactly this long after the fault
# (`ProtectionIED.run`: `reportEventAt(last + 0.5)`, then `+ 0.6`).
RECOVERY_OFFSET_S = 0.1

# Transfer-time requirements for trip messages (IEC 61850-5, type 1A).
TIMING_LIMITS_S = (0.003, 0.010)

DEFAULT_T_RELAY = 0.020
DEFAULT_T_BREAKER = 0.050
DEFAULT_T_BACKUP = 0.400
DEFAULT_T_BACKUP_SENSITIVITY = (0.250, 0.600)


class ConsequenceError(ValueError):
    """The input cannot support the event reconstruction."""


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family_of(run_id):
    """`DETERMINISTIC_BURST-l100-b3-s20260101` -> `DETERMINISTIC_BURST`.

    Per-row `attack_variant` reads `none` on every unlabeled row of an attack
    run, so the run's family has to come from its identifier.
    """
    return str(run_id).split("-", 1)[0]


def state_table(frames):
    """One row per (trace, StNum): whether it carried a trip, its t, first arrival."""
    missing = [c for c in COLUMNS if c not in frames.columns]
    if missing:
        raise ConsequenceError("dataset lacks column(s): %s" % ", ".join(missing))
    grouped = frames.groupby(["trace_id", "StNum"], sort=True)
    states = grouped.agg(
        run_id=("run_id", "first"),
        cb=("cbStatus", "max"),
        t=("t", "min"),
        first_arrival=("GooseTimestamp", "min"),
    ).reset_index()
    return states


def fault_parity(trace_states):
    """The StNum parity of fault states in one trace.

    Observed from the states that carry `cbStatus = 1` when any survived;
    otherwise implied by the first state, which is a recovery-type state, so
    faults are the states one change after it.
    """
    faults = trace_states.loc[trace_states["cb"] == 1, "StNum"].to_numpy()
    if len(faults):
        parities = np.bincount(faults.astype(np.int64) % 2, minlength=2)
        return int(np.argmax(parities))
    return int((trace_states["StNum"].min() + 1) % 2)


def fault_events(states):
    """Every fault event a trace should contain, delivered or not.

    Returns one row per event with the event time `t_f`, whether a fault frame
    was delivered, the earliest delivered fault frame, and whether the event
    is observable at all.
    """
    rows = []
    for trace_id, trace_states in states.groupby("trace_id", sort=True):
        by_st = trace_states.set_index("StNum")
        lo, hi = int(by_st.index.min()), int(by_st.index.max())
        parity = fault_parity(trace_states)
        run_id = trace_states["run_id"].iloc[0]
        first = lo + 1 if (lo + 1) % 2 == parity else lo + 2
        for st in range(first, hi, 2):
            fault = by_st.loc[st] if st in by_st.index else None
            delivered = fault is not None and fault["cb"] == 1
            if delivered:
                t_f = float(fault["t"])
                arrival = float(fault["first_arrival"])
            elif st + 1 in by_st.index:
                t_f = float(by_st.loc[st + 1, "t"]) - RECOVERY_OFFSET_S
                arrival = np.nan
            else:
                t_f = np.nan
                arrival = np.nan
            rows.append((run_id, trace_id, st, t_f, delivered, arrival))
    events = pd.DataFrame(rows, columns=["run_id", "trace_id", "StNum", "t_f",
                                         "delivered", "first_arrival"])
    events["observable"] = events["t_f"].notna()
    return events


def score_events(events, t_relay, t_breaker, t_backup):
    """Layer 1 and layer 2 per event."""
    out = events.copy()
    out["missed"] = out["observable"] & ~out["delivered"]
    out["latency"] = np.where(out["delivered"], out["first_arrival"] - out["t_f"], np.nan)
    for limit in TIMING_LIMITS_S:
        key = "violates_%gms" % (limit * 1000)
        out[key] = out["observable"] & (out["missed"] | (out["latency"] > limit))
    ideal = t_relay + t_breaker
    out["t_clear"] = np.where(out["delivered"], t_relay + out["latency"] + t_breaker,
                              np.where(out["observable"], t_backup + t_breaker, np.nan))
    out["relative_energy"] = out["t_clear"] / ideal
    return out


def per_run(scored):
    """Sums per run - the resampling unit."""
    observable = scored[scored["observable"]]
    violation_cols = [c for c in scored.columns if c.startswith("violates_")]
    agg = {"events": ("StNum", "size"), "missed": ("missed", "sum"),
           "energy_sum": ("relative_energy", "sum")}
    for col in violation_cols:
        agg[col] = (col, "sum")
    runs = observable.groupby("run_id").agg(**agg)
    runs["unobservable"] = (scored[~scored["observable"]].groupby("run_id").size()
                            .reindex(runs.index).fillna(0).astype(int))
    return runs


def bootstrap_rate(numer, denom, iterations, rng, confidence=0.95):
    """Percentile interval of sum(numer)/sum(denom) resampling runs."""
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    n = len(numer)
    if n == 0 or denom.sum() == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, n, size=(iterations, n))
    stats = numer[idx].sum(axis=1) / np.maximum(denom[idx].sum(axis=1), 1e-300)
    alpha = (1 - confidence) / 2
    return (float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha)))


def summarise(scored, run_meta, iterations, seed, keys):
    """One row per group in `keys`, with run-bootstrap intervals."""
    runs = per_run(scored).join(run_meta, how="left")
    rng = np.random.default_rng(seed)
    rows = []
    for group, block in runs.groupby(list(keys), sort=True, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        ev = block["events"].sum()
        row = dict(zip(keys, group))
        row.update({
            "runs": int(len(block)),
            "events": int(ev),
            "unobservable": int(block["unobservable"].sum()),
            "missed_rate": float(block["missed"].sum() / ev) if ev else np.nan,
            "missed_ci": bootstrap_rate(block["missed"], block["events"], iterations, rng),
            "relative_energy": float(block["energy_sum"].sum() / ev) if ev else np.nan,
            "relative_energy_ci": bootstrap_rate(block["energy_sum"], block["events"],
                                                 iterations, rng),
        })
        for col in [c for c in block.columns if c.startswith("violates_")]:
            row[col] = float(block[col].sum() / ev) if ev else np.nan
        rows.append(row)
    return rows


def latency_quantiles(scored, run_meta, keys):
    """Latency among delivered trips, per group."""
    delivered = scored[scored["delivered"]].join(run_meta, on="run_id")
    out = []
    for group, block in delivered.groupby(list(keys), sort=True, dropna=False):
        group = group if isinstance(group, tuple) else (group,)
        lat = block["latency"].to_numpy()
        row = dict(zip(keys, group))
        row.update({"delivered": int(len(lat)),
                    "late_first_frame": float(np.mean(lat > TIMING_LIMITS_S[1])),
                    "p50_ms": float(np.median(lat) * 1000),
                    "p95_ms": float(np.quantile(lat, 0.95) * 1000),
                    "max_ms": float(lat.max() * 1000)})
        out.append(row)
    return out


def run_metadata(frames):
    meta = frames.groupby("run_id").agg(burst_size=("burst_size", "first"),
                                        loss_rate=("loss_rate", "first"))
    meta["family"] = [family_of(r) for r in meta.index]
    return meta


def analyse(frames, t_relay=DEFAULT_T_RELAY, t_breaker=DEFAULT_T_BREAKER,
            t_backup=DEFAULT_T_BACKUP, sensitivity=DEFAULT_T_BACKUP_SENSITIVITY,
            iterations=1000, seed=42):
    states = state_table(frames)
    events = fault_events(states)
    meta = run_metadata(frames)
    scored = score_events(events, t_relay, t_breaker, t_backup)
    result = {
        "by_family": summarise(scored, meta, iterations, seed, ("family",)),
        "by_cell": summarise(scored, meta, iterations, seed,
                             ("family", "burst_size", "loss_rate")),
        "latency_by_family": latency_quantiles(scored, meta, ("family",)),
        "sensitivity": {},
    }
    for backup in sensitivity:
        alt = score_events(events, t_relay, t_breaker, backup)
        result["sensitivity"]["%g" % backup] = summarise(alt, meta, iterations, seed,
                                                         ("family",))
    return result


def fmt_ci(ci, digits=3):
    lo, hi = ci
    if np.isnan(lo):
        return "—"
    return "[%.*f, %.*f]" % (digits, lo, digits, hi)


def build_markdown(result, source, digest, params):
    lines = ["# Protection consequence - result",
             "",
             "- Generated: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
             "- Dataset: `%s` (SHA-256 `%s`)" % (os.path.basename(source), digest[:16]),
             "- `T_relay` %.0f ms, `T_breaker` %.0f ms, `T_backup` %.0f ms "
             "(sensitivity %s ms); bootstrap over runs, %d replicates, seed %d"
             % (params["t_relay"] * 1000, params["t_breaker"] * 1000,
                params["t_backup"] * 1000,
                ", ".join("%.0f" % (b * 1000) for b in params["sensitivity"]),
                params["iterations"], params["seed"]),
             "- Design and expected readings: `protection_consequence.md` §§1-6.",
             "",
             "## By family",
             "",
             "| Family | runs | events | unobservable | missed trip [95% CI] | "
             "> 3 ms | > 10 ms | relative energy [95% CI] |",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for r in result["by_family"]:
        lines.append("| `%s` | %d | %s | %s | %.4f %s | %.4f | %.4f | %.2f %s |" % (
            r["family"], r["runs"], f"{r['events']:,}", f"{r['unobservable']:,}",
            r["missed_rate"], fmt_ci(r["missed_ci"], 4), r["violates_3ms"],
            r["violates_10ms"], r["relative_energy"], fmt_ci(r["relative_energy_ci"], 2)))
    lines += ["", "## Latency among delivered trips", "",
              "| Family | delivered | first frame > 10 ms late | p50 ms | p95 ms | max ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in result["latency_by_family"]:
        lines.append("| `%s` | %s | %.4f | %.1f | %.1f | %.1f |" % (
            r["family"], f"{r['delivered']:,}", r["late_first_frame"], r["p50_ms"],
            r["p95_ms"], r["max_ms"]))
    lines += ["", "## By cell", "",
              "| Family | burst | loss | runs | events | missed trip [95% CI] | "
              "> 10 ms | relative energy |",
              "|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in result["by_cell"]:
        lines.append("| `%s` | %s | %s | %d | %s | %.4f %s | %.4f | %.2f |" % (
            r["family"], r["burst_size"], r["loss_rate"], r["runs"], f"{r['events']:,}",
            r["missed_rate"], fmt_ci(r["missed_ci"], 4), r["violates_10ms"],
            r["relative_energy"]))
    lines += ["", "## Sensitivity to `T_backup` (relative energy by family)", ""]
    backups = list(result["sensitivity"])
    lines.append("| Family | " + " | ".join("%s ms" % (float(b) * 1000) for b in backups)
                 + " | %.0f ms (primary) |" % (params["t_backup"] * 1000))
    lines.append("|---|" + "---:|" * (len(backups) + 1))
    primary = {r["family"]: r["relative_energy"] for r in result["by_family"]}
    for fam in primary:
        vals = [next(r["relative_energy"] for r in result["sensitivity"][b]
                     if r["family"] == fam) for b in backups]
        lines.append("| `%s` | " % fam + " | ".join("%.2f" % v for v in vals)
                     + " | %.2f |" % primary[fam])
    return "\n".join(lines) + "\n"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    return obj


def load_frames(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=COLUMNS)
    return pd.read_csv(path, usecols=COLUMNS)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Prepared Parquet or CSV dataset.")
    parser.add_argument("--t-relay", type=float, default=DEFAULT_T_RELAY)
    parser.add_argument("--t-breaker", type=float, default=DEFAULT_T_BREAKER)
    parser.add_argument("--t-backup", type=float, default=DEFAULT_T_BACKUP)
    parser.add_argument("--t-backup-sensitivity", type=float, nargs="*",
                        default=list(DEFAULT_T_BACKUP_SENSITIVITY))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Markdown report path.")
    parser.add_argument("--json-out", default=None, help="JSON report path.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = {"t_relay": args.t_relay, "t_breaker": args.t_breaker,
              "t_backup": args.t_backup, "sensitivity": args.t_backup_sensitivity,
              "iterations": args.iterations, "seed": args.seed}
    try:
        digest = sha256_file(args.dataset)
        frames = load_frames(args.dataset)
        result = analyse(frames, args.t_relay, args.t_breaker, args.t_backup,
                         args.t_backup_sensitivity, args.iterations, args.seed)
    except (OSError, ConsequenceError, ValueError, KeyError) as exc:
        print("PROTECTION CONSEQUENCE FAILED\n%s" % exc, file=sys.stderr)
        return 2
    text = build_markdown(result, args.dataset, digest, params)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(to_jsonable({"dataset": args.dataset, "dataset_sha256": digest,
                                   "params": params, "result": result}), fh, indent=2)
            fh.write("\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
