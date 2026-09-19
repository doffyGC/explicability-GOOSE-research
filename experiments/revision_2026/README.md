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
| `generate_run_matrix.py` | Executes the preregistered variant × seed × loss-rate × burst-size matrix in the patched ERENO. Writes one CSV and one provenance sidecar per independent run plus `run_matrix.json`; restores `params.properties` even after interruption. `--legitimate-stream {exclude,include}` (default `exclude`) decides whether the legitimate publisher's complete stream is written alongside the attacker's forwarded copies; `include` reproduces the 265-run pool's setting, whose interleaving fills every gap a grayhole makes and gives every attack row a byte-identical `normal` twin (`label_duplication_audit.md`). The choice is recorded in `run_matrix.json`, because two pools generated under different settings are otherwise indistinguishable from their CSVs and are not comparable. | A.3 |
| `test_generate_run_matrix.py` | Pins the two `attacks.properties` flags the generator writes: `attacks.orientedGrayhole` tracks the family, and `attacks.legitimate` defaults to off for **both** families. Getting the second one wrong is not a visible failure — the generator succeeds and the CSVs look ordinary — so it is asserted rather than left to a code read. Also covers appending the key when an older checkout lacks it, leaving unrelated flags alone, and `set_property` surviving regex metacharacters in a Windows path. | A.3 |
| `merge_runs.py` | Pools the per-run CSVs from `generate_run_matrix.py` into one dataset, validating each run before it is allowed in: one run per file, `run_id`/`seed` agreeing across rows, sidecar and filename, no repeated run identity, and no two runs sharing a payload. Runs are streamed one at a time, so peak memory is one run. Writes `merge_report.md`. Derives nothing — `event_id` and `split_group` stay with `add_experiment_metadata.py`. **Its payload fingerprint includes `class`**, so two runs that are byte-identical in content under *different* labels pass it — which is how the `FRG`/`CONGESTION_LOSS` collision got through (`benign_controls.md` §8). A cross-run, label-blind collision gate is an open task. | A.3 |
| `add_experiment_metadata.py` | Adds the experimental-unit columns to the dataset. Derives `event_id` from the GOOSE `(StNum, t)` state key, groups events into `trace_id`/`run_id`, maps `class` to `attack_variant`, reads the ERENO generation parameters (`seed`, `loss_rate`, `burst_size`, `traffic_rate`, `substation_config`) from a JSON manifest, and sets `split_group`. Writes `metadata_audit.md` recording what was derived, what is still missing and how many independent units actually exist. On a native-schema Parquet input (one row group per run, as `merge_runs.py` writes), processes and writes one row group at a time instead of loading the whole dataset (`run_native_chunked`) — ~8.8× lower peak RSS, same output. | A.3 |
| `check_no_leakage.py` | Validates versioned JSON/CSV grouped splits before training. Fails on train/test group overlap, unknown/omitted groups, duplicate assignments or invalid fold coverage; can write a machine-readable audit. | A.4 |
| `test_check_no_leakage.py` | Positive and deliberately leaking fixtures for the split-integrity checker, including its command-line interface and exit codes. | A.4 |
| `data_card.md` | Documents generation, labels, features, experimental units, legacy provenance limits, intended/prohibited uses, hashes and the release procedure for regenerated runs. | A.5 |
| `prepare_grouped_dataset.py` | Sorts messages inside each trace and recomputes ERENO's delta features without crossing trace boundaries; drops the predecessor-less first row of every trace and writes a hash-bound audit. Processes trace-aligned Parquet inputs one row group at a time (~10.2× lower peak RSS); rejects, rather than silently mis-computing, any file whose row groups are not trace-aligned. | B.2 |
| `generate_grouped_splits.py` | Creates StratifiedGroupKFold, GroupKFold, LeaveOneGroupOut or LOETO folds; checks class coverage and persists exact groups as JSON plus CSV before invoking the independent leakage checker. | B.1, B.3, B.4 |
| `run_grouped_validation.py` | Trains only from persisted, hash-bound grouped splits and writes fold metrics plus row/group-linked predictions. Supports a clearly marked capped technical smoke — when capped and given a Parquet dataset, samples row group by row group instead of loading the full dataset first (~8.4× lower peak RSS on the smoke path). The uncapped/full training path uses `load_grouped_arrays`, which fills a preallocated `float32` array straight from the Parquet row groups and never builds a DataFrame of features at all: the old route held three simultaneous copies (the `pd.concat` chunk list plus its result, then `feature_matrix`'s `drop`) and peaked at **10.02 GB** on the 265-run/23.2M-row pool, which stopped fitting on a 15.6 GB machine — every full run died during the load, whatever model followed. The streaming loader peaks at **4.03 GB** against 3.46 GB of unavoidable feature matrix. Values are identical, not merely equivalent: `test_validation_protocol.py` asserts X/y/group codes/row index/column order/label ordering element for element against the DataFrame path, and both full runs reproduced their card-D.3 `grouped_predictions.csv` byte for byte. The CSV and technical-smoke paths still go through `load_frame`/`prepare_arrays`. `--balance {none,downsample,smote}` (checklist E) rebalances only each fold's TRAIN partition — via `imbalanced-learn` — and always evaluates on the untouched original-distribution test fold; see `validation_protocol.md`, "Balancing scenarios". `--save-scores` additionally writes `grouped_scores.parquet`, the per-row class posteriors behind `y_pred` - without them the run is pinned to the one operating point its training prior implies and no threshold question can be answered (`grouped_pr_curves.py`); `grouped_predictions.csv` is byte-identical with and without the flag. `--model {decision-tree,xgboost,random-forest,logistic-regression}` is checklist D.3's untuned family comparison, and `--max-train-rows-per-fold` its documented per-fold train cap — proportionally stratified by (`split_group`, `class`), train-only, so the run stays a `full_grouped_run`; see `ablations_baselines.md` §7. `--feature-set` is checklist D.1's feature-group ablation: each named set removes exactly one group of `FEATURE_GROUPS` (`no-sequence` removes two disjoint ones), the groups partition all 40 model features, and the report records `feature_set`/`feature_groups_dropped`/`features_dropped` plus `features_in_no_group` — the survivors no group claims, so a column added after the registry was written is visible rather than silently outside the partition. Resolved against the dataset's own columns and **fatal** on any mismatch, including a partially present group: an ablation that drops nothing produces a run identical to the reference and reads exactly like "this group does not matter". See `ablations_baselines.md` §13. `run_folds` also takes a `model_selector` hook, which is card D.4's injection point: left unset every run behaves exactly as before, and `run_nested_tuning.py` uses it so a tuned run inherits this loop's invariants instead of a second copy of them. | B.1, B.5, D.1, D.3, D.4, E |
| `test_validation_protocol.py` | Tests trace-boundary deltas, grouped split integrity, unseen-class blocking, dataset/report/split hash binding, train-only rebalancing, the D.3 per-fold train cap (proportions preserved, rare strata never emptied, reproducible per seed), the model-family registry being left at library defaults, and `load_grouped_arrays` matching the DataFrame path element for element (values, dtypes, column order, class and group label ordering) plus its refusals: missing group column, a non-numeric feature, no features left. | B.1–B.5, D.3, E |
| `validation_protocol.md` | Defines the canonical section B workflow, smoke evidence and the blockers separating implementation readiness from publishable evaluation. | B.1–B.5 |
| `benign_controls.md` | Defines the section C plan: taxonomy of the 7 benign-degradation mechanisms, attack pairing rules, label vocabulary (`class=benign_degradation` + `impairment_mode`), matrix design, pipeline-integration constraints and per-milestone status tracking. | C.1–C.5 |
| `ablations_baselines.md` | Defines the section D plan: the recorded scope decision (minimum defensible scope, 2026-09-12), what is deferred and why (top-k SHAP ablation → card F; temporal model), execution order, per-item cost estimates calibrated on the existing full runs, the invariants every card-D run must respect and a status tracker. §7 records the measured per-fold train subsampling policy (why only Random Forest needs a cap, and the 4M-row cap plus its decision-tree control run); §8 the D.3 run matrix and the champion criterion fixed before execution. | D.1–D.5 |
| `run_rule_baseline.py` | Checklist D.2's threshold baseline: the detector the learned model has to earn its complexity against. One rule is one feature and one comparison, with the threshold fitted **only on each fold's train partition** of the same persisted `splits_grouped.json` the learned runs consume, chosen to maximise `ANY_ATTACK` F1 (accuracy would select "never fire" at 2% prevalence). `RULES` holds the six arms: the three interval/timing rules `ablations_baselines.md` §15 measured as the carriers, plus the `sqNum`/`stNum` gap detectors and the delay threshold §2 preregistered — a preregistered arm is kept and reported rather than dropped because a later result predicts it will lose. Writes exactly what `run_grouped_validation.py` writes, so `check_prediction_integrity.py`, `bootstrap_run_intervals.py` and `grouped_pr_curves.py` audit the baseline under the same invariants as the model. That contract is multiclass and a rule is binary: `y_true` keeps the dataset's six classes untouched, a firing row is reported as the fold's train-majority attack class (recorded per fold) and a non-firing row as `normal`, so **only the `ANY_ATTACK` axis of such a run is a result** — its macro F1 is floored by construction, since the rule cannot name a family. The persisted score is the train partition's ECDF rescaled to cross 0.5 exactly at the threshold: strictly monotone in the feature, so it moves no ranking and therefore no AP, and `argmax` reproduces the rule's own decision. | D.2 |
| `test_rule_baseline.py` | Tests the baseline the paper is compared against: F1-not-accuracy calibration at low prevalence, exhaustive vs. quantised threshold search, a non-finite row counting as a miss rather than an absence, tie-breaking towards the quieter detector, the score being fitted on the given train values (the leak that would make the baseline unbeatable), monotonicity, `argmax(posterior)` reproducing the decision at float32 on both sides of the threshold, the designated class coming from train attack rows only, and the CLI end to end — including a full run through `check_prediction_integrity.py` with 0 failures, and the refusals on a stale dataset hash and a missing rule column. | D.2 |
| `run_nested_tuning.py` | Checklist D.4's nested hyperparameter search: does the champion earn its hyperparameters, or is a library default being reported as a capability claim? Hyperparameters are chosen on `StratifiedGroupKFold` folds drawn from the **outer fold's train groups only**, grouped by `run_id` exactly as the outer protocol is, then refit on the full outer train partition and scored once on outer test groups nothing in the selection has seen — selecting on the outer test fold is the same leak this revision exists to remove, moved one level up. It owns no fold loop: it injects a selector into `run_grouped_validation.run_folds` (the `model_selector` hook), so a tuned run inherits that loop's group-overlap, empty-partition, test-only-class and train-only-balancing refusals rather than a second copy of them, and writes the same three artifacts every audit already consumes plus a `tuning` block and a per-fold `selection` record. Selection is on **AP over `ANY_ATTACK`**, not argmax macro F1, per `ablations_baselines.md` §11; macro F1 is recorded beside it for every point so the disagreement between the two criteria is a column rather than an argument. `GRIDS` holds the preregistered 12-point `champion-xgboost` grid whose axes were read off the champion's error structure (§17) — **point 0 is the library default**, verified bit for bit against `classifier("xgboost", ...)`, so "tuning bought nothing" is readable off the selection table instead of inferred, and ties go to the earlier point so a tie never reports as a gain. `--inner-max-rows` is §2's documented subsample, proportional inside every (`split_group`, `class`) stratum so all 212 training runs stay represented; `--plan-only` prints the fit multiplier without touching the dataset. | D.4 |
| `test_nested_tuning.py` | Tests the two ways a tuning run is silently worthless. **It selects on rows it is scored on**: the search's fits are handed an array where every outer test row carries a sentinel value, and no fitted block may contain it — structural, not a call-graph read. **Its "default" point is not the default**: XGBoost's sklearn wrapper leaves unset parameters `None` and applies the booster's defaults in C++, so both estimators are fitted and required to return bit-identical posteriors. Also covers the inner split never crossing a group, every inner row tested once, too few groups being fatal rather than a silent reduction, AP matching `sklearn` on the pooled attack score, an unrankable fold scoring NaN rather than 0, ties keeping the default, a grid nothing can be scored on being fatal, and the CLI end to end — including a full run through `check_prediction_integrity.py` with 0 failures, a stale dataset hash, a grid refusing a family it was not designed for, and `--plan-only` writing nothing. | D.4 |
| `benign_confusion_report.py` | Reads a `run_grouped_validation.py` run's predictions, rejoins `impairment_mode` from the hash-verified prepared dataset, and reports a 6x6 (or gracefully reduced) confusion matrix separating `normal` from `benign_degradation`, a per-mode outcome breakdown, and attack_fpr/alert_rate per mode contrasted against ideal-normal and in-run-baseline-normal traffic. Writes `benign_confusion.md`. | C.4 |
| `check_prediction_integrity.py` | Reconciles one or more finished grouped-validation runs before any statistical test is written: every `row_index` predicted exactly once (row-level counterpart to `check_no_leakage.py`), per-fold predictions == `test_rows` == sum of per-class support, class sums matching both the report and the hash-verified dataset, and every metric recomputed from a `numpy.bincount` confusion matrix — accuracy (micro), macro *and* weighted averages, per class, each explicitly labelled — then cross-checked against what the run recorded. When the run carries `grouped_scores.parquet`, also re-derives `argmax(posterior)` over **every** scored row and requires it to reproduce `y_pred` - the runner only verifies its own argmax on the first block of each fold, and a self-check on a fifth of the rows is not an audit. The one exception is exact: on a fold whose report says it fell back to `model.predict`, `y_pred` is a float64 argmax against the persisted float32 scores, so a mismatch is forgiven only where the two leading posteriors are bit-identical. Across runs, reports whether predictions are pairable (same rows, same ground truth) and the 2×2 agreement table a McNemar-style test consumes. Non-zero exit on any failure. | E.4, E.5 |
| `test_prediction_integrity.py` | Tests the audit: metric recomputation against `sklearn`'s `classification_report` (including the zero-division path), duplicate/gap row coverage, support and dataset class-sum disagreements, pairable vs. non-pairable runs, the CLI's exit codes on tampered supports/metrics, and the scores reconciliation (absent file, one flipped argmax, a posterior block that is not a distribution, missing scored rows, disagreeing ground truth, and the float32-tie exception for a fold that fell back to `model.predict` - forgiven on a tie, still a failure on a strict winner or outside a fallback fold). | D.5, E.4, E.5 |
| `bootstrap_run_intervals.py` | Confidence intervals that resample **runs** (`split_group`), not rows and not folds: rows inside a run are correlated by construction, so a row-level bootstrap reports intervals far narrower than the evidence supports, and 5 folds are a partition of the runs rather than 5 independent estimates. Each replicate redraws the runs with replacement, sums their per-run confusion matrices and recomputes every metric. A class carried by fewer than 2 runs is reported as `not estimable` rather than given the zero-width interval resampling would produce, and classes under 20 runs are flagged as thin. Also reports **paired** differences against the first run — both models scored on the same resampled runs, so the variation they share cancels — because two overlapping marginal intervals do not mean two models are indistinguishable. Writes `run_bootstrap.md`. | D.5, E.4 |
| `test_bootstrap_run_intervals.py` | Tests the bootstrap: metric recomputation against `sklearn`, undefined-vs-zero handling for absent classes, per-run splitting of a predictions CSV across chunk boundaries, wider intervals for a class concentrated in few runs, the zero-width collapse that motivates the `not estimable` flag, a present-but-never-predicted class scoring 0 (not NaN) so macro F1 matches `sklearn`, paired separation where marginal intervals overlap, realignment by group label, and the CLI's exit codes. | D.5, E.4 |
| `check_label_duplication.py` | Checks the invariant nothing else in the chain owns: that the label is a function of the features. Reports rows sharing a message key, rows whose content is repeated under a different label, attack rows with a content-identical non-attack row, and the subset identical on *every* model feature (irreducible - no classifier can separate them). Also names **which columns differ between two copies of one message**, which is what distinguishes a writer emitting twice from two genuinely different messages. Runs on raw per-run ERENO CSVs or on a pooled dataset; non-zero exit on any conflict, so it can gate a regeneration the way `check_no_leakage.py` gates training. Found the defect in `label_duplication_audit.md`; reports **zero findings** on the corrected pool. Intra-run only — it does not see two *different* runs sharing a payload under different labels (`benign_controls.md` §8). | A.5, B.5 |
| `test_check_label_duplication.py` | Tests that the audit separates redundancy (a message repeated under the *same* label - not a defect) from the real conflict, sees through the delta columns that differ on any duplicate by construction, counts irreducible pairs apart from merely content-identical ones, reports `benign_degradation` conflicts that the attack-only count misses, keeps string columns (MACs, `gocbRef`) inside row identity, and exits 1/2 correctly. | A.5, B.5 |
| `feature_signal_probe.py` | Scores a run's model, each single feature and any trivial `--rule` on one average-precision axis against the same positives, so "where is the signal" is measured rather than argued. Written as the control for `label_duplication_audit.md`: the natural inference from that audit - that the model is just reading the duplicate marker - is **false**, and the probe is what shows it (`sqDiff != 0` scores exactly chance; the model scores 7.8x). Also D.1's premise in miniature: the lift is multivariate, so no single-feature ranking could find it. | D.1, A.5 |
| `label_duplication_audit.md` | The finding, its root cause in ERENO's scenario (a *dropping* attack modelled as an *emitting* one), what it invalidates and what does not, and the experimental-unit question it forces before any regeneration. | A.5, B.5 |
| `grouped_pr_curves.py` | Recovers the threshold axis every other report collapses. `run_grouped_validation.py` records `argmax(p)`, so each published number is one point on a curve fixed by the *training* prior - which is why `none` (attack recall ~0) and `downsample` (recall ~0.70 at a ~39% false-positive rate) are the same score read at two thresholds ~213x apart, not two verdicts on detectability. Reads the persisted posteriors and reports grid-quantised average precision per class (threshold-free, against the prevalence floor a random detector scores) plus operating points at a fixed **alert budget**, with the false alarms broken down by the class that produced them. Thresholds are calibrated on the *other* folds, never on the fold being scored; `--prior auto` divides out a balanced run's training prior and leaves an unbalanced one alone, which is what makes differently balanced runs comparable in one invocation. Intervals resample runs, reusing `bootstrap_run_intervals.py`'s unit and its `not estimable` floor, and several `--run`s are compared **paired** — both scored on the same redrawn runs, each keeping its own thresholds — because overlapping marginal intervals settle nothing. Writes `pr_curves.md`. | D.5 |
| `test_grouped_pr_curves.py` | Tests the curves against `sklearn`'s `average_precision_score`, the prevalence floor a zero-information detector scores, the budget threshold's loosest-that-fits rule and its unreachable-budget case, the prior correction being a no-op on an unbalanced run and exact on a balanced one, `ScoreWriter`/`labels_from_proba` round-tripping and agreeing with `model.predict`, and - by construction, on a fold whose scores would pick a wildly different threshold - that a threshold is never selected on the fold it scores. The paired bootstrap is tested on a strictly better ranker (must separate) and against a run paired with itself (must give exactly zero — which is what fails if the two sides are resampled independently). | D.5 |
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

> **This subsection is a historical record of the audit that motivated
> regenerating ERENO's output — it describes the legacy `gray-GOOSE.csv` and
> the unpatched generator, not the current state.** All three gaps below
> (unseeded RNG, one variant per build, no run/batch identifiers) have since
> been closed in `../ereno` (branch `refactor/fix-major-revision`): `run.seed`
> now threads through `Rng` into every `Random` construction on the generation
> path; `attack.orientedGrayhole.variant/.discardRate/.burstSize` are read
> from `params.properties` per run (`RunContext.loadConfigs()`); and
> `RunContext.csvHeader()`/`csvRow()` emit `run_id`, `trace_id`, `seed`,
> `loss_rate`, `burst_size` and (as of card C1) `impairment_*` natively. See
> `data_card.md` §4 for how this maps onto the experimental-unit hierarchy,
> and `run_matrix_plan.json` for the resulting 120-run attack matrix (executed
> in full and merged with the 85-run benign matrix — see `validation_protocol.md`,
> "Full-scale results").

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
