"""
Generate a matrix of independent ERENO runs.

Checklist item A.3, and the reviewers' P0 on dataset validation: the submitted
dataset had four traces, one per attack class, produced by hand-editing
`OrientedGrayHoleCreator` and rebuilding once per variant. Four groups, each
tied to a single class, cannot support grouped cross-validation: leaving one
group out leaves out a whole class.

This script drives the patched ERENO (see the `run.*` and
`attack.orientedGrayhole.*` keys in `params.properties`) once per cell of a
variant x seed x loss-rate x burst-size matrix, so every cell is a run with its
own identity written into every row. Each run lands in its own CSV next to its
`.run.json` sidecar; `merge_runs.py` pools them.

It edits `params.properties` in the ERENO checkout and restores it afterwards,
including on failure or Ctrl-C.

Usage
-----
    # what would run, without running it
    python experiments/revision_2026/generate_run_matrix.py --dry-run

    # the real thing (hours; start with --smoke to check the wiring)
    python experiments/revision_2026/generate_run_matrix.py --smoke
    python experiments/revision_2026/generate_run_matrix.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_ERENO = os.path.abspath(os.path.join(REPO_ROOT, "..", "ereno"))

SCENARIO_CLASS = "br.ufu.facom.ereno.scenarios.BalancedSamambaiaScenario"

VARIANTS = ["DETERMINISTIC_BURST", "FULLY_RANDOMIZED", "RANDOMIC_BURST", "RANDOMIC_MESSAGE"]

# Revision matrix: five independent seeds and two sensitivity axes around the
# submitted operating point (loss_rate=15%, burst_size=5).  A run targets 1,000
# malicious messages: calibration showed that inheriting ERENO's old target of
# 100,000 would turn this matrix into hundreds of GB and weeks of computation.
DEFAULT_SEEDS = [20260101, 20260102, 20260103, 20260104, 20260105]
DEFAULT_LOSS_RATES = [5, 15, 30]
DEFAULT_BURST_SIZES = [3, 5, 10]
DEFAULT_TARGET_MALICIOUS = 1000
DEFAULT_BATCH_SIZE = 9000
DEFAULT_MAX_ITERATIONS = 500

# A cell small enough to check the wiring end to end in a couple of minutes.
SMOKE = {
    "seeds": [20260101, 20260102],
    "loss_rates": [15],
    "burst_sizes": [5],
    "target_malicious": 300,
    "batch_size": 3000,
    "max_iterations": 20,
}


def set_property(text, key, value):
    """Replace `key=...` in a .properties file, or append it if absent."""
    pattern = re.compile(r"^%s=.*$" % re.escape(key), re.MULTILINE)
    if pattern.search(text):
        # A callable replacement keeps Java escapes such as ``\\u00e7`` from
        # being interpreted as regular-expression replacement escapes.
        return pattern.sub(lambda _: "%s=%s" % (key, value), text)
    return text.rstrip("\n") + "\n%s=%s\n" % (key, value)


def java_property_value(value):
    """Escape non-ASCII text for Java's ISO-8859-1 Properties reader.

    This matters on Windows when the checkout path contains characters such as
    the `ç` in `Cybersegurança`: writing raw UTF-8 makes Java resolve a different
    path even though Python can open it normally.
    """
    return "".join(ch if ord(ch) < 128 else "\\u%04x" % ord(ch) for ch in str(value))


def read(path):
    # newline="" preserves the checkout's original LF/CRLF convention when
    # params.properties is restored after a run.
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def run_cmd(cmd, cwd, log_path):
    """Run a command, tee-ing its output to a log file. Returns the exit code."""
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode


def build_matrix(args):
    cells = []
    for variant in args.variants:
        for seed in args.seeds:
            for loss in args.loss_rates:
                for burst in args.burst_sizes:
                    # FULLY_RANDOMIZED drops independent single messages, so
                    # burst_size is fixed to its effective value 1.
                    if variant == "FULLY_RANDOMIZED" and burst != args.burst_sizes[0]:
                        continue
                    # DETERMINISTIC_BURST drops on every state change, so its
                    # effective loss rate is 100% and discardRate is inactive.
                    if variant == "DETERMINISTIC_BURST" and loss != args.loss_rates[0]:
                        continue
                    effective_loss = 100 if variant == "DETERMINISTIC_BURST" else loss
                    effective_burst = 1 if variant == "FULLY_RANDOMIZED" else burst
                    cells.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "loss_rate": effective_loss,
                            "burst_size": effective_burst,
                            "discard_rate_config": loss,
                            "burst_size_config": burst,
                            "run_id": "%s-l%d-b%d-s%d" % (
                                variant, effective_loss, effective_burst, seed
                            ),
                        }
                    )
    run_ids = [c["run_id"] for c in cells]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Matrix produced duplicate run_id values")
    return cells


def matrix_summary(cells):
    """Return counts that make the experimental design auditable."""
    by_variant = {}
    for cell in cells:
        by_variant[cell["variant"]] = by_variant.get(cell["variant"], 0) + 1
    return {
        "total_runs": len(cells),
        "runs_by_variant": by_variant,
        "distinct_seeds": len({c["seed"] for c in cells}),
        "effective_loss_rates": sorted({c["loss_rate"] for c in cells}),
        "effective_burst_sizes": sorted({c["burst_size"] for c in cells}),
    }


def write_plan(path, args, cells):
    plan = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "purpose": "Independent grouped-validation runs for the Gray-GOOSE major revision",
        "design": {
            "seeds": args.seeds,
            "loss_rates_config": args.loss_rates,
            "burst_sizes_config": args.burst_sizes,
            "target_malicious_per_run": args.target_malicious,
            "batch_size": args.batch_size,
            "max_iterations": args.max_iterations,
            "inactive_dimensions": {
                "DETERMINISTIC_BURST": "discard_rate (effective loss_rate=100)",
                "FULLY_RANDOMIZED": "burst_size (effective burst_size=1)",
            },
        },
        "summary": matrix_summary(cells),
        "runs": cells,
    }
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(plan, fh, indent=2)
        fh.write("\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ereno-dir", default=DEFAULT_ERENO, help="ERENO checkout (default: sibling ../ereno).")
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "runs"), help="Where run CSVs land.")
    p.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--loss-rates", nargs="+", type=int, default=DEFAULT_LOSS_RATES)
    p.add_argument("--burst-sizes", nargs="+", type=int, default=DEFAULT_BURST_SIZES)
    p.add_argument("--target-malicious", type=int, default=DEFAULT_TARGET_MALICIOUS,
                   help="Attack messages targeted per run (default: %(default)s).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help="Messages generated per iteration (default: %(default)s).")
    p.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS,
                   help="Safety limit per run (default: %(default)s).")
    p.add_argument("--plan-out", help="Write the complete planned matrix to JSON.")
    p.add_argument("--smoke", action="store_true", help="Tiny matrix and tiny runs, to check the wiring.")
    p.add_argument("--dry-run", action="store_true", help="Print the matrix and exit.")
    p.add_argument("--skip-existing", action="store_true", help="Leave runs whose CSV already exists.")
    args = p.parse_args(argv)

    if args.smoke:
        args.seeds = SMOKE["seeds"]
        args.loss_rates = SMOKE["loss_rates"]
        args.burst_sizes = SMOKE["burst_sizes"]
        args.target_malicious = SMOKE["target_malicious"]
        args.batch_size = SMOKE["batch_size"]
        args.max_iterations = SMOKE["max_iterations"]

    for name in ("target_malicious", "batch_size", "max_iterations"):
        if getattr(args, name) <= 0:
            p.error("--%s must be positive" % name.replace("_", "-"))

    ereno = os.path.abspath(args.ereno_dir)
    params_path = os.path.join(ereno, "src", "main", "resources", "params.properties")
    if not os.path.exists(params_path):
        raise SystemExit("params.properties not found under %s - pass --ereno-dir." % ereno)

    cells = build_matrix(args)
    summary = matrix_summary(cells)
    print("Matrix: %d runs | target: %d attack messages/run" %
          (len(cells), args.target_malicious))
    print("By variant: " + ", ".join("%s=%d" % item for item in summary["runs_by_variant"].items()))
    for c in cells:
        print("  %s" % c["run_id"])
    if args.plan_out:
        write_plan(args.plan_out, args, cells)
        print("Plan written: %s" % os.path.abspath(args.plan_out))
    if args.dry_run:
        return 0

    out_dir = os.path.abspath(args.out_dir)
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    original = read(params_path)
    results = []
    try:
        for i, cell in enumerate(cells, 1):
            csv_path = os.path.join(out_dir, cell["run_id"] + ".csv")
            if args.skip_existing and os.path.exists(csv_path):
                print("[%d/%d] %s - exists, skipping" % (i, len(cells), cell["run_id"]))
                results.append(dict(cell, status="skipped", csv=csv_path))
                continue

            print("[%d/%d] %s" % (i, len(cells), cell["run_id"]), flush=True)

            text = original
            text = set_property(text, "run.seed", cell["seed"])
            text = set_property(text, "run.id", cell["run_id"])
            text = set_property(text, "run.traceId", cell["run_id"])
            text = set_property(text, "run.scenarioId", "SC-%s-l%d-b%d" % (cell["variant"], cell["loss_rate"], cell["burst_size"]))
            text = set_property(text, "attack.orientedGrayhole.variant", cell["variant"])
            text = set_property(text, "attack.orientedGrayhole.discardRate", cell["discard_rate_config"])
            text = set_property(text, "attack.orientedGrayhole.burstSize", cell["burst_size_config"])
            text = set_property(
                text, "scenario.path",
                java_property_value(out_dir.replace("\\", "/") + "/"),
            )
            text = set_property(text, "scenario.datasetName", cell["run_id"])
            if args.target_malicious is not None:
                text = set_property(text, "scenario.targetMaliciousMessages", args.target_malicious)
            if args.batch_size is not None:
                text = set_property(text, "scenario.batchSize", args.batch_size)
            if args.max_iterations is not None:
                text = set_property(text, "scenario.maxIterations", args.max_iterations)
            write(params_path, text)

            # The writer appends, so a leftover file from a previous attempt
            # would silently concatenate two runs into one CSV.
            for stale in (csv_path, csv_path + ".run.json"):
                if os.path.exists(stale):
                    os.remove(stale)

            log = os.path.join(log_dir, cell["run_id"] + ".log")
            mvn = "mvn.cmd" if os.name == "nt" else "mvn"
            rc = run_cmd([mvn, "-q", "compile"], ereno, log)
            if rc != 0:
                print("    compile failed - see %s" % log)
                results.append(dict(cell, status="compile_failed", csv=None))
                continue

            rc = run_cmd(["java", "-cp", os.path.join("target", "classes"), SCENARIO_CLASS], ereno, log)
            status = "ok" if rc == 0 and os.path.exists(csv_path) else "failed"
            rows = None
            if status == "ok":
                with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
                    rows = sum(1 for _ in fh) - 1
                print("    %s rows" % format(rows, ","))
            else:
                print("    failed - see %s" % log)
            results.append(dict(cell, status=status, csv=csv_path if status == "ok" else None, rows=rows))
    finally:
        write(params_path, original)
        print("Restored %s" % params_path)

    manifest = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "ereno_dir": ereno,
        "scenario_class": SCENARIO_CLASS,
        "design": {
            "target_malicious_per_run": args.target_malicious,
            "batch_size": args.batch_size,
            "max_iterations": args.max_iterations,
        },
        "summary": matrix_summary(cells),
        "runs": results,
    }
    manifest_path = os.path.join(out_dir, "run_matrix.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    print("\n%d/%d runs available. Manifest: %s" % (ok, len(results), manifest_path))
    if ok < len(results):
        print("Some runs failed; inspect %s before pooling." % log_dir)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
