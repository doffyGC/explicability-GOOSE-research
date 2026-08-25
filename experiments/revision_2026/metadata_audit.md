# Dataset metadata audit - Gray-GOOSE

- Generated: 2026-08-24 18:50:22 UTC
- Source dataset: `data\CSV files\gray-GOOSE.csv`
- Rows: 1,006,989
- Output: (audit only, nothing written)
- Manifest: NONE - generation parameters left null
- split_group level: `run`

## 1. Assumption check: is each attack class a single stream?

The trace derivation is only valid if, within an attack class, `StNum -> t`
is a strictly increasing bijection (one publisher stream).

| attack_variant | rows | distinct StNum | StNum with >1 t | share | t monotonic in StNum | single stream |
|---|---:|---:|---:|---:|:--:|:--:|
| SAG.DB | 102,470 | 102,470 | 0 | 0.0000% | yes | yes |
| FRG | 102,236 | 38,211 | 5 | 0.0131% | yes | yes |
| SAG.PB | 100,373 | 100,373 | 0 | 0.0000% | yes | yes |
| SAG.PBM | 101,910 | 98,941 | 0 | 0.0000% | yes | yes |

A few duplicated states (below 0.1% of a class) are treated as
generator noise: the same StNum re-published ~0.1 s later with overlapping SqNum.
They are a data-quality point worth a line in the data card, not a second stream.

## 2. Event and trace attribution

- Distinct GOOSE events, keyed `(StNum, t)`: **575,698**
- Events anchored directly by an attack row: **339,958**
- Events resolved after bracket propagation: **369,774** (64.2%)
- Events left ambiguous (`T-UNRESOLVED`): **205,924** (35.8%)
- Events carrying two different attack classes (conflicts): **21**

Propagation passes:

| pass | events gained | events assigned |
|---:|---:|---:|
| 0 | 29,618 | 369,576 |
| 1 | 187 | 369,763 |
| 2 | 7 | 369,770 |
| 3 | 2 | 369,772 |
| 4 | 2 | 369,774 |
| 5 | 0 | 369,774 |

Rows per trace:

| trace_id | rows | share |
|---|---:|---:|
| T-UNRESOLVED | 331,167 | 32.9% |
| T00-SAG.DB | 182,100 | 18.1% |
| T01-FRG | 124,782 | 12.4% |
| T02-SAG.PB | 196,652 | 19.5% |
| T03-SAG.PBM | 172,288 | 17.1% |

> **Independent experimental units available: 4.**
> `T-UNRESOLVED` is not a sixth unit - it is a pool of messages from all
> traces that the CSV cannot separate, so it cannot go on one side of a split.
> That leaves 4 usable groups covering 67.1% of the rows, one per attack scenario.
> A GroupKFold over this many groups gives very few, highly correlated folds,
> and each group is tied to exactly one attack class, so a left-out group is a
> left-out *class* - the fold has no positive examples of what it is tested on.
> Regenerating the dataset with several ERENO runs per scenario (different
> seeds, durations and traffic profiles) is the only way to get the number of
> independent units the grouped-statistics part of the revision needs.

## 3. Substation configuration

- Identity columns inspected: `gocbRef`, `datSet`, `goID`, `gooseAppid`, `ethSrc`, `ethDst`
- Distinct publisher configurations found: **1**

> The whole dataset uses one publisher (`SUB-A`): a single GOOSE control
> block, dataset, appID and MAC pair. `substation_config` is therefore constant
> and carries no information. This is the 'traffic diversity' gap the reviewers
> raised, and it cannot be fixed by annotation - only by regeneration.

## 4. Generation parameters (manifest-supplied)

`seed`, `loss_rate`, `burst_size` and `traffic_rate` are ERENO *inputs*. They
cannot be recovered from a message table and are only written when a manifest
supplies them.

| column | non-null rows | status |
|---|---:|---|
| scenario_id | 1,006,989 | derived fallback (`SC-<trace scenario>`) |
| seed | 0 | **MISSING - fill the manifest or regenerate** |
| loss_rate | 0 | **MISSING - fill the manifest or regenerate** |
| burst_size | 0 | **MISSING - fill the manifest or regenerate** |
| traffic_rate | 0 | **MISSING - fill the manifest or regenerate** |

## 5. Empirical diagnostics (NOT written into the dataset)

Measurements of the delivered stream, useful for sanity-checking a manifest.
A measured rate is not the generation parameter - do not copy these into
`loss_rate` / `burst_size` / `traffic_rate`. The `T-UNRESOLVED` row mixes
several traces, so its figures describe the pool, not any one run.

| trace_id | messages | events | span (s) | msgs/s | states with a gap | mean states dropped | max states dropped | variants |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T-UNRESOLVED | 331,167 | 190,354 | 7,390,045.3 | 0.0448 | 0.7881 | 3.641 | 5661 | FRG, SAG.DB, SAG.PB, SAG.PBM, none |
| T00-SAG.DB | 182,100 | 102,456 | 2,061,714.1 | 0.0883 | 1.0 | 1.0 | 3 | SAG.DB, none |
| T01-FRG | 124,782 | 41,416 | 628,050.5 | 0.1987 | 0.4199 | 1.204 | 8 | FRG, none |
| T02-SAG.PB | 196,652 | 114,662 | 7,390,061.0 | 0.0266 | 0.9632 | 5.631 | 73 | SAG.PB, none |
| T03-SAG.PBM | 172,288 | 111,229 | 3,139,429.1 | 0.0549 | 0.6711 | 2.689 | 34 | SAG.PBM, none |

## 6. What still blocks checklist item A.3

- [ ] `seed`, `loss_rate`, `burst_size`, `traffic_rate` per scenario (ERENO config).
- [ ] `substation_config` diversity: the dataset has a single publisher.
- [ ] 205,924 events cannot be attributed to a trace from the CSV alone.
- [ ] Enough independent runs for grouped CV and per-group confidence intervals.

All four are properties of the generation run. Annotating the delivered CSV
cannot create them; regenerating with ERENO while emitting the identifiers
can, and is the intent of Phase 1 of the revision plan.
