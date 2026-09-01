# experiments/revision_2026

Scripts produced for the major revision of Gray-GOOSE paper.

## Current system (baseline, frozen for reference)

Main scripts of the pipeline that generated the submitted (rejected)
results, kept here as a reference point for the revision.

| Script | Description |
|---|---|
| `main.py` | Entry point: orchestrates the full pipeline — load dataset, preprocess, train (K-Fold CV), evaluate, save reports, run SHAP. |
| `config.py` | Global settings: seed, K-Fold splits, model type/params, discarded/delta feature columns, class names, dataset path, SHAP plot list. |
| `data/loader.py` | Loads the dataset from `.csv`/`.parquet` into a DataFrame. |
| `data/preprocess.py` | Splits the DataFrame into features (`X`) and target (`y`), label-encodes the class column. |
| `model/train.py` | Trains a classifier (XGBoost/RF/SVM/MLP/decision tree/logistic regression) with `StratifiedKFold` CV, plus a final model on all data. |
| `model/evaluate.py` | Computes per-class/global metrics with confidence intervals across CV folds, aggregates the confusion matrix, saves Markdown/log reports and `y_true`/`y_pred` arrays (for McNemar). |
| `model/matrix_confusion.py` | Plots and saves the confusion matrix (log-scale heatmap) as SVG. |
| `explainability/shap_analysis.py` | Runs SHAP (`TreeExplainer`) on the last CV fold's validation set and saves bar/beeswarm plots per class. |
| `notebooks/experiment.ipynb` | Same pipeline as `main.py`, runnable interactively section-by-section; used to reproduce and inspect results manually. |

> Known issues this revision must fix: `StratifiedKFold` splits at message level
> (no `run_id`/`trace_id` grouping → leakage risk), no benign-only packet-loss
> controls, SHAP computed on a CV fold rather than a proper held-out set, and
> McNemar/statistics run over pooled message-level predictions. See
> `prioridades_revisao_gray_goose.pdf` at the repo root for the full plan.

## New scripts for this revision

Each script added here should get one line below describing what it does
and which checklist item it covers. Keep entries short; details belong in
the script's own docstring/comments, not in this file.

| Script | Description | Checklist ref. |
|---|---|---|
| `generate_run_matrix.py` | Executes the preregistered variant × seed × loss-rate × burst-size matrix in the patched ERENO. Writes one CSV and one provenance sidecar per independent run plus `run_matrix.json`; restores `params.properties` even after interruption. | A.3 |
| `merge_runs.py` | Pools the per-run CSVs from `generate_run_matrix.py` into one dataset, validating each run before it is allowed in: one run per file, `run_id`/`seed` agreeing across rows, sidecar and filename, no repeated run identity, and no two runs sharing a payload. Runs are streamed one at a time, so peak memory is one run. Writes `merge_report.md`. Derives nothing — `event_id` and `split_group` stay with `add_experiment_metadata.py`. | A.3 |
| `add_experiment_metadata.py` | Adds the experimental-unit columns to the dataset. Derives `event_id` from the GOOSE `(StNum, t)` state key, groups events into `trace_id`/`run_id`, maps `class` to `attack_variant`, reads the ERENO generation parameters (`seed`, `loss_rate`, `burst_size`, `traffic_rate`, `substation_config`) from a JSON manifest, and sets `split_group`. Writes `metadata_audit.md` recording what was derived, what is still missing and how many independent units actually exist. | A.3 |
| `check_no_leakage.py` | Validates versioned JSON/CSV grouped splits before training. Fails on train/test group overlap, unknown/omitted groups, duplicate assignments or invalid fold coverage; can write a machine-readable audit. | A.4 |
| `test_check_no_leakage.py` | Positive and deliberately leaking fixtures for the split-integrity checker, including its command-line interface and exit codes. | A.4 |
| `data_card.md` | Documents generation, labels, features, experimental units, legacy provenance limits, intended/prohibited uses, hashes and the release procedure for regenerated runs. | A.5 |
| `prepare_grouped_dataset.py` | Sorts messages inside each trace and recomputes ERENO's delta features without crossing trace boundaries; drops the predecessor-less first row of every trace and writes a hash-bound audit. | B.2 |
| `generate_grouped_splits.py` | Creates StratifiedGroupKFold, GroupKFold, LeaveOneGroupOut or LOETO folds; checks class coverage and persists exact groups as JSON plus CSV before invoking the independent leakage checker. | B.1, B.3, B.4 |
| `run_grouped_validation.py` | Trains only from persisted, hash-bound grouped splits and writes fold metrics plus row/group-linked predictions. Supports a clearly marked capped technical smoke. | B.1, B.5 |
| `test_validation_protocol.py` | Tests trace-boundary deltas, grouped split integrity, unseen-class blocking and dataset/report/split hash binding. | B.1–B.5 |
| `validation_protocol.md` | Defines the canonical section B workflow, smoke evidence and the blockers separating implementation readiness from publishable evaluation. | B.1–B.5 |
| `benign_controls.md` | Defines the section C plan: taxonomy of the 7 benign-degradation mechanisms, attack pairing rules, label vocabulary (`class=benign_degradation` + `impairment_mode`), matrix design, pipeline-integration constraints and per-milestone status tracking. | C.1–C.5 |
| `benign_confusion_report.py` | Reads a `run_grouped_validation.py` run's predictions, rejoins `impairment_mode` from the hash-verified prepared dataset, and reports a 6x6 (or gracefully reduced) confusion matrix separating `normal` from `benign_degradation`, a per-mode outcome breakdown, and attack_fpr/alert_rate per mode contrasted against ideal-normal and in-run-baseline-normal traffic. Writes `benign_confusion.md`. | C.4 |
| `test_benign_controls.py` | Tests the card-C pipeline wiring: `benign_degradation` in `VARIANT_OF_CLASS`, `SC-BENIGN_*` event-type inference, `impairment_*` columns excluded from both the feature matrix and the payload fingerprint, closed-set LOETO over the benign families (contrasted with the open-set attack-variant case), and `benign_confusion_report.py`'s matrix/per-mode/attack_fpr computations and hash-bound dataset rejoin. | C.3, C.4 |

```bash
# what the delivered CSV can and cannot support, no files written
python experiments/revision_2026/add_experiment_metadata.py --audit-only

# skeleton for the generation parameters that are not in the CSV
python experiments/revision_2026/add_experiment_metadata.py --write-manifest-template

# annotate (drop --manifest to leave the generation parameters null)
python experiments/revision_2026/add_experiment_metadata.py \
    --manifest experiments/revision_2026/manifest.json
```

Regenerated runs need no manifest: the patched ERENO writes the generation
parameters into the rows, so the annotation script detects them and passes them
through instead of reconstructing anything.

### Regeneration matrix

The versioned design is in `run_matrix_plan.json`. Its defaults are:

| Axis | Values |
|---|---|
| variants | `DETERMINISTIC_BURST`, `FULLY_RANDOMIZED`, `RANDOMIC_BURST`, `RANDOMIC_MESSAGE` |
| seeds | `20260101` to `20260105` (5 independent runs per cell) |
| configured loss rates | 5%, 15%, 30% |
| configured burst sizes | 3, 5, 10 messages |
| target | 1,000 malicious messages per run |

Inactive dimensions are not duplicated: `DETERMINISTIC_BURST` always has an
effective loss rate of 100%, while `FULLY_RANDOMIZED` always has an effective
burst size of 1. The resulting matrix has **120 independent runs**: 15 DB,
15 FRG, 45 PB and 45 PBM. The submitted operating point (15%, burst 5) is
included, with neighbouring values providing sensitivity analysis.

```bash
# inspect/recreate the versioned plan without executing ERENO
python experiments/revision_2026/generate_run_matrix.py --dry-run \
    --plan-out experiments/revision_2026/run_matrix_plan.json

# execute; safe to resume after interruption
python experiments/revision_2026/generate_run_matrix.py --skip-existing
```

```bash
# validate the runs without writing anything
python experiments/revision_2026/merge_runs.py --check-only

# pool them, then derive event_id / split_group
python experiments/revision_2026/merge_runs.py
python experiments/revision_2026/add_experiment_metadata.py \
    --dataset data/runs/gray-GOOSE-runs.parquet
```

Grouped splits use either JSON with `split_id`, `train_groups` and
`test_groups`, or long-form CSV with `split_id,partition,split_group`.

```bash
# must pass before model training
python experiments/revision_2026/check_no_leakage.py \
    --dataset data/runs/gray-GOOSE-runs-metadata.parquet \
    --splits experiments/revision_2026/splits.json \
    --report experiments/revision_2026/leakage_audit.json

# regression tests, including intentional leakage
python -m unittest discover -s experiments/revision_2026 \
    -p "test_check_no_leakage.py" -v
```

`manifest.json` holds the generation parameters recovered by reading the ERENO
source that produced the dataset, each traced to a file and line. What it can
and cannot supply is recorded below.

### A.3 status after auditing the ERENO source

| Column | Status | Source |
|---|---|---|
| `event_id`, `attack_variant` | recovered | derived from GOOSE `(StNum, t)` / `class` |
| `trace_id`, `run_id`, `split_group` | partial | 4 runs; 33% of rows unattributable |
| `scenario_id` | recovered | `OrientedGrayHoleCreator:29` (hardcoded enum) |
| `loss_rate` | recovered | `OrientedGrayHoleCreator:27` → `discardRate = 15` |
| `burst_size` | recovered | `OrientedGrayHoleCreator:28` → `toDiscardPackets = 5` |
| `traffic_rate` | recovered | `params.properties` → `goose.timing.maxTime=1000` |
| `substation_config` | recovered but degenerate | `params.properties`; one publisher for every run |
| `seed` | **does not exist** | `IED.java:45,56` → `new Random(System.nanoTime())` |

> **`seed` is unrecoverable, not merely unrecorded.** ERENO builds a new
> `Random` from the wall clock on every call, so no run can be reproduced.
> Confirmed empirically: re-running the same `FULLY_RANDOMIZED` configuration
> produced `ereno/src/datasets/todos_os_ataques.csv`, which shares only **14 of
> 102,236** `(StNum, t)` keys with the FRG trace in `gray-GOOSE.csv` —
> statistically identical, element-wise unrelated.

Two further gaps found in the generator, both fixable only by regeneration:

- **Batch boundaries are never written.** `BalancedSamambaiaScenario` loops
  `runDevicesBatch(batchSize=90000)` until `targetMaliciousMessages=100000`, so
  each run is many batches — a real run boundary that `CSVWritter` does not emit.
- **One variant per build.** The attack variant is a hardcoded field
  (`OrientedGrayHoleCreator:29`), so each of the four classes came from a
  separate hand-edited build. That is why there are only four traces.

To close A.3 properly the generator needs: a seeded RNG threaded through
`IED.randomBetween`; variant, `discardRate` and `toDiscardPackets` moved to
`params.properties`; and `CSVWritter` emitting run/batch/scenario/seed columns.
Then a matrix of runs (variants × loss rates × seeds) gives grouped CV real
independent units.

As scripts land (e.g. leakage checks, grouped-split protocol, benign
control generation, ablations, baselines, balancing experiments,
grouped statistics / held-out SHAP), add a row here in the same commit.
