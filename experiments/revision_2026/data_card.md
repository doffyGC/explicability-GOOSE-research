# Gray-GOOSE dataset card

## 1. Summary and status

Gray-GOOSE is a synthetic smart-grid network dataset generated with ERENO. It
combines IEC 61850 GOOSE station-bus messages, correlated Sampled Values (SV)
and electrical measurements. Its research purpose is to evaluate detection of
state-aware grayhole (SAG) packet-dropping attacks.

This card distinguishes two artifacts that must not be presented as equivalent:

| Artifact | Status | Appropriate use |
|---|---|---|
| `gray-GOOSE.csv` | Dataset used by the submitted paper; available locally | Audit and baseline reproduction only. Its run identity and random state were not recorded. |
| `gray-GOOSE-metadata.parquet` | Legacy dataset annotated after generation | Data-quality analysis only. Some metadata is recovered or derived, not generator-emitted. |
| Regenerated run matrix | **Planned and smoke-tested, not generated in full** | Intended dataset for the revised grouped evaluation. The complete design is versioned in `run_matrix_plan.json`. |

The legacy artifact must not be used to claim leakage-free grouped validation.
The regenerated artifact must not be described as available until all planned
runs have passed `merge_runs.py`, annotation and `check_no_leakage.py`.

## 2. Artifact identity

| File | Rows | Size | SHA-256 |
|---|---:|---:|---|
| `data/CSV files/gray-GOOSE.csv` | 1,006,989 | 588,751,063 bytes | `B77E6EF58DE054336DBFA19B3C9469AEE56C859B695B139029C9801FC79B32AE` |
| `data/CSV files/gray-GOOSE-metadata.parquet` | 1,006,989 | 66,461,377 bytes | `FE7F7F12B674267A9F404A6CC6E3BCC3BB4B9A159C0A0BC84001E131BC3C7BCD` |

Hashes identify the local artifacts audited for this revision. Regenerating or
rewriting either file requires updating this section and `metadata_audit.md`.

No dataset license is declared in this repository. Confirm ownership and add an
explicit license before public redistribution. The absence of a declared
license is not permission to redistribute.

## 3. Generation environment

### Legacy artifact

- Generator: ERENO, scenario class `BalancedSamambaiaScenario`.
- Intended malicious-message target: 100,000 per attack scenario.
- Batch size: 90,000 messages; maximum 1,000 iterations.
- Publisher: one GOOSE control block (`LD/LLN0$GO$gcbA`), one dataset
  (`LD/LLN0$IntLockA`), `goID=IntLockA`, `appId=0x00003001`.
- Network endpoints: source `01:0c:cd:01:2f:78`, destination
  `01:0c:cd:01:2f:77`.
- GOOSE timing: 100 ms minimum, 1,000 ms steady-state maximum, 11,000 ms TTL.
- Event retransmission uses exponential backoff; simulated network delay is
  uniformly drawn from 1 to 31 ms.
- One publisher/substation configuration is present in every scenario.

The legacy ERENO called `new Random(System.nanoTime())` for individual random
draws. There is no run seed to recover: `seed=null` is a factual value, not
missing annotation. Re-executing the same configuration produces statistically
similar but element-wise different traces.

### Revised generator and planned matrix

ERENO commit `7e6eb7d` introduces one seeded RNG, generator-emitted run metadata,
configurable attack parameters and per-run JSON sidecars. Three executions of
the same smoke cell (`FULLY_RANDOMIZED`, seed `20260101`) produced byte-identical
CSVs with MD5 `42f78a780335b473ce5323698329aa4c`; seed `20260102` produced a
different hash. This establishes deterministic reproduction for the patched
single-threaded generation path.

The planned matrix contains 120 independent runs:

| Dimension | Values |
|---|---|
| Attack variants | DB, FRG, PB, PBM |
| Seeds | 20260101, 20260102, 20260103, 20260104, 20260105 |
| Configured loss rates | 5%, 15%, 30% |
| Configured burst sizes | 3, 5, 10 messages |
| Target | 1,000 labelled attack messages per run |

`DETERMINISTIC_BURST` does not use configured loss probability and reports
effective `loss_rate=100`. `FULLY_RANDOMIZED` drops messages independently and
reports effective `burst_size=1`. Redundant cells along those inactive axes are
not generated. The full preregistered list is `run_matrix_plan.json`.

## 4. Instances, events and experimental units

One row represents one retained GOOSE message correlated with an SV cycle and
electrical features. A dropped packet is absent from the dataset. Attack labels
are attached by ERENO to an observable retained message associated with the
drop behavior; they must not be interpreted as packet-level ground truth for a
packet that was removed.

The hierarchy used by the revised evaluation is:

| Identifier | Meaning |
|---|---|
| `event_id` | One GOOSE state/event, keyed within a trace by `(StNum, t)`; retransmissions share the event. |
| `trace_id` | One publisher stream. The current regenerated design has one trace per run. |
| `run_id` | One independently seeded ERENO execution for one parameter cell. |
| `split_group` | Unit kept intact across train/test; defaults to `run_id`. |
| `batch_index` | Internal generation batch within a run; not an independent run. |

Messages, retransmissions and events from one `split_group` must never occur on
both sides of a train/test split. `check_no_leakage.py` enforces this invariant.

## 5. Labels and attack definitions

| Raw `class` | `attack_variant` | Description |
|---|---|---|
| `normal` | `none` | Retained message not labelled as an attack consequence. It may come from a run configured with an attack. |
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | `SAG.DB` | On each state change, drops a deterministic burst; effective trigger loss is 100%. |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | `FRG` | Evaluates each message independently against the configured loss probability. |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | `SAG.PB` | On a state change, probabilistically triggers a consecutive burst of configured length. |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | `SAG.PBM` | Within a bounded post-state-change window, probabilistically evaluates messages for dropping. |

`attack_variant` describes the row label, not merely the configuration of the
run. Consequently benign rows use `none`; the configured run scenario remains
available in `scenario_id` and `run_id`.

Legacy class counts are:

| Class | Rows |
|---|---:|
| normal | 600,000 |
| SAG.DB | 102,470 |
| FRG | 102,236 |
| SAG.PB | 100,373 |
| SAG.PBM | 101,910 |

These counts describe messages, not independent observations.

## 6. Features

The base table contains the following feature families:

- instantaneous and RMS three-phase current/voltage measurements;
- trapezoidal electrical-area aggregates;
- GOOSE protocol fields, including `StNum`, `SqNum`, timestamps, TTL, frame and
  APDU sizes, identifiers and status flags;
- SV/GOOSE timing and protocol-correlation features;
- delta features such as sequence, timestamp, length and status differences;
- target column `class`.

The revised metadata adds `run_id`, `trace_id`, `event_id`, `scenario_id`,
`seed`, `attack_variant`, `loss_rate`, `burst_size`, `traffic_rate`,
`substation_config` and `split_group`. Native regenerated CSVs additionally
carry `batch_index`; grouped preparation adds `message_index` to preserve the
emitted sequence inside each trace.

Delta features in the submitted table were generated before trustworthy trace
boundaries existed. Revised experiments must recompute deltas after sorting
inside each trace and must never difference the last message of one trace with
the first message of another.

## 7. Legacy provenance audit

The delivered CSV is shuffled and contains no native run/trace identifiers.
Post-hoc reconstruction found:

- 575,698 distinct `(StNum, t)` events;
- 369,774 events attributable to one attack trace after propagation;
- 205,924 events (35.8%) ambiguous;
- 331,167 rows (32.9%) in `T-UNRESOLVED`;
- 21 event keys carrying conflicting attack classes;
- four usable inferred attack traces, each tied to one attack class;
- one publisher/substation configuration.

`T-UNRESOLVED` is a mixed pool, not an independent fifth run, and must not be
placed wholesale on either side of a split. Leaving out one of the four inferred
groups also leaves out its attack class. Therefore the legacy artifact cannot
support the grouped protocol required by the revision.

Generation parameters recovered from source inspection are recorded in
`manifest.json`. They are configuration evidence, not row-level provenance.
The annotated legacy Parquet must retain this distinction.

## 8. Recommended uses

- Audit the submitted results and document why message-level splitting leaks.
- Develop and test feature-processing, merge and split-integrity tooling.
- With the completed regenerated matrix: grouped classification, per-run or
  cluster-aware statistics, held-out SHAP and parameter sensitivity analysis.
- Compare attack variants only when all metrics are aggregated by independent
  run/fold rather than treating messages as independent samples.

## 9. Uses to avoid

- Do not use random message-level train/test splitting.
- Do not use `T-UNRESOLVED` as an independent run.
- Do not impute or invent seeds for the legacy dataset.
- Do not infer configured packet-loss probability from observed class ratios.
- Do not call SHAP values causal physical explanations.
- Do not claim operational validation, topology diversity or field deployment:
  the data are synthetic and use one publisher configuration.
- Do not claim robustness to benign congestion, jitter, delay, reordering,
  duplication or random loss until those controls are generated and evaluated.
- Do not use this dataset alone to assess compliance with IEC 62351-6 or the
  effectiveness of a complete defense-in-depth architecture.

## 10. Validation and release procedure

Before a regenerated dataset is used in an experiment:

1. Generate each cell with `generate_run_matrix.py`; retain its CSV, log and
   `.run.json` sidecar.
2. Run `merge_runs.py`. It rejects inconsistent names/rows/sidecars, repeated
   identities and duplicate payloads.
3. Run `add_experiment_metadata.py` to derive `event_id` and `split_group`.
4. Generate and version the grouped splits as JSON/CSV.
5. Run `check_no_leakage.py`; training must stop on a non-zero exit code.
6. Record output hashes, row/class/run counts, ERENO commit and matrix plan.
7. Update this card from “planned” to “released” only after all checks pass.

Recommended split command:

```bash
python experiments/revision_2026/check_no_leakage.py \
  --dataset data/runs/gray-GOOSE-runs-metadata.parquet \
  --splits experiments/revision_2026/splits.json \
  --report experiments/revision_2026/leakage_audit.json
```

The default policy expects GroupKFold/LeaveOneGroupOut semantics: every split
partitions all groups, and every group appears in test exactly once across the
fold collection. Explicit flags relax these constraints for a documented
holdout or repeated-CV design; they never permit train/test overlap.

## 11. Privacy, safety and maintenance

The dataset is synthetic and contains no intended personal data. MAC addresses,
IED/control-block names and electrical traces are simulated configuration data,
not captured production traffic. Nevertheless, generated attack traces can aid
security research and should be shared with context that discourages claims of
field realism beyond the modeled scenario.

Maintainers of the revised artifact should version the generator commit, matrix
plan, merge report, metadata audit, leakage audit and this card together. Any
change to labels, topology, timing, seed set, loss rates or burst sizes creates
a new dataset version and requires new hashes and counts.

## 12. Known gaps at this revision stage

- The 120-run matrix is designed but has not been executed completely.
- No benign degradation controls have been generated yet.
- The current matrix still uses one publisher/substation configuration.
- The legacy dataset license is undeclared.
- Grouped split generation and validation have passed on a six-run technical
  smoke. Final five-fold splits have not been produced because the complete
  120-run matrix does not exist yet.
