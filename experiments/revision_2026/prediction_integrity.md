# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-12 17:02:26 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | folds | rows |
|---|---|---|---|---:|---:|
| `grouped-validation-full` | full_grouped_run | decision-tree | none | 5 | 20,796,921 |
| `grouped-validation-full-downsample` | full_grouped_run | decision-tree | downsample | 5 | 20,796,921 |
| `grouped-validation-full-smote` | full_grouped_run | decision-tree | smote | 5 | 20,796,921 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `grouped-validation-full`

30 checks, 0 failed.

<details><summary>All checks passed - expand for detail</summary>

| check | expected | observed | result |
|---|---|---|---|
| each row_index predicted at most once | 0 duplicates | 0 duplicates | pass |
| prediction rows == report rows_used | 20796921 | 20796921 | pass |
| full run covers dataset rows 0..N-1 with no gaps | 0..20796920 | 0..20796920 over 20796921 rows | pass |
| fold-00: predictions == test_rows == sum(per-class support) | 4716325 == 4716325 | 4716325 predictions | pass |
| fold-01: predictions == test_rows == sum(per-class support) | 5582606 == 5582606 | 5582606 predictions | pass |
| fold-02: predictions == test_rows == sum(per-class support) | 4236815 == 4236815 | 4236815 predictions | pass |
| fold-03: predictions == test_rows == sum(per-class support) | 2502792 == 2502792 | 2502792 predictions | pass |
| fold-04: predictions == test_rows == sum(per-class support) | 3758383 == 3758383 | 3758383 predictions | pass |
| class sum (predictions vs. report support): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 17094 | 17094 | pass |
| class sum (predictions vs. report support): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 21436 | 21436 | pass |
| class sum (predictions vs. report support): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 | 46959 | pass |
| class sum (predictions vs. report support): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 | 54828 | pass |
| class sum (predictions vs. report support): benign_degradation | 270680 | 270680 | pass |
| class sum (predictions vs. report support): normal | 20385924 | 20385924 | pass |
| class sum (predictions vs. dataset): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 17094 in dataset | 17094 predicted | pass |
| class sum (predictions vs. dataset): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 21436 in dataset | 21436 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 in dataset | 46959 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 in dataset | 54828 predicted | pass |
| class sum (predictions vs. dataset): benign_degradation | 270680 in dataset | 270680 predicted | pass |
| class sum (predictions vs. dataset): normal | 20385924 in dataset | 20385924 predicted | pass |
| fold-00: recorded accuracy matches recomputation | 0.987345443751 | 0.987345443751 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.266570309458 | 0.266570309458 | pass |
| fold-01: recorded accuracy matches recomputation | 0.991661242079 | 0.991661242079 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.312134197122 | 0.312134197122 | pass |
| fold-02: recorded accuracy matches recomputation | 0.985178253004 | 0.985178253004 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.277688959121 | 0.277688959121 | pass |
| fold-03: recorded accuracy matches recomputation | 0.973625854646 | 0.973625854646 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.261737131411 | 0.261737131411 | pass |
| fold-04: recorded accuracy matches recomputation | 0.986145105488 | 0.986145105488 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.260803886658 | 0.260803886658 | pass |

</details>

### `grouped-validation-full-downsample`

30 checks, 0 failed.

<details><summary>All checks passed - expand for detail</summary>

| check | expected | observed | result |
|---|---|---|---|
| each row_index predicted at most once | 0 duplicates | 0 duplicates | pass |
| prediction rows == report rows_used | 20796921 | 20796921 | pass |
| full run covers dataset rows 0..N-1 with no gaps | 0..20796920 | 0..20796920 over 20796921 rows | pass |
| fold-00: predictions == test_rows == sum(per-class support) | 4716325 == 4716325 | 4716325 predictions | pass |
| fold-01: predictions == test_rows == sum(per-class support) | 5582606 == 5582606 | 5582606 predictions | pass |
| fold-02: predictions == test_rows == sum(per-class support) | 4236815 == 4236815 | 4236815 predictions | pass |
| fold-03: predictions == test_rows == sum(per-class support) | 2502792 == 2502792 | 2502792 predictions | pass |
| fold-04: predictions == test_rows == sum(per-class support) | 3758383 == 3758383 | 3758383 predictions | pass |
| class sum (predictions vs. report support): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 17094 | 17094 | pass |
| class sum (predictions vs. report support): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 21436 | 21436 | pass |
| class sum (predictions vs. report support): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 | 46959 | pass |
| class sum (predictions vs. report support): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 | 54828 | pass |
| class sum (predictions vs. report support): benign_degradation | 270680 | 270680 | pass |
| class sum (predictions vs. report support): normal | 20385924 | 20385924 | pass |
| class sum (predictions vs. dataset): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 17094 in dataset | 17094 predicted | pass |
| class sum (predictions vs. dataset): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 21436 in dataset | 21436 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 in dataset | 46959 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 in dataset | 54828 predicted | pass |
| class sum (predictions vs. dataset): benign_degradation | 270680 in dataset | 270680 predicted | pass |
| class sum (predictions vs. dataset): normal | 20385924 in dataset | 20385924 predicted | pass |
| fold-00: recorded accuracy matches recomputation | 0.545757766905 | 0.545757766905 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.202142402762 | 0.202142402762 | pass |
| fold-01: recorded accuracy matches recomputation | 0.558658089072 | 0.558658089072 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.204966839720 | 0.204966839720 | pass |
| fold-02: recorded accuracy matches recomputation | 0.524415864275 | 0.524415864275 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.204858981772 | 0.204858981772 | pass |
| fold-03: recorded accuracy matches recomputation | 0.441600021096 | 0.441600021096 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.200140965270 | 0.200140965270 | pass |
| fold-04: recorded accuracy matches recomputation | 0.496356544823 | 0.496356544823 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.172889275658 | 0.172889275658 | pass |

</details>

### `grouped-validation-full-smote`

30 checks, 0 failed.

<details><summary>All checks passed - expand for detail</summary>

| check | expected | observed | result |
|---|---|---|---|
| each row_index predicted at most once | 0 duplicates | 0 duplicates | pass |
| prediction rows == report rows_used | 20796921 | 20796921 | pass |
| full run covers dataset rows 0..N-1 with no gaps | 0..20796920 | 0..20796920 over 20796921 rows | pass |
| fold-00: predictions == test_rows == sum(per-class support) | 4716325 == 4716325 | 4716325 predictions | pass |
| fold-01: predictions == test_rows == sum(per-class support) | 5582606 == 5582606 | 5582606 predictions | pass |
| fold-02: predictions == test_rows == sum(per-class support) | 4236815 == 4236815 | 4236815 predictions | pass |
| fold-03: predictions == test_rows == sum(per-class support) | 2502792 == 2502792 | 2502792 predictions | pass |
| fold-04: predictions == test_rows == sum(per-class support) | 3758383 == 3758383 | 3758383 predictions | pass |
| class sum (predictions vs. report support): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 17094 | 17094 | pass |
| class sum (predictions vs. report support): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 21436 | 21436 | pass |
| class sum (predictions vs. report support): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 | 46959 | pass |
| class sum (predictions vs. report support): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 | 54828 | pass |
| class sum (predictions vs. report support): benign_degradation | 270680 | 270680 | pass |
| class sum (predictions vs. report support): normal | 20385924 | 20385924 | pass |
| class sum (predictions vs. dataset): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 17094 in dataset | 17094 predicted | pass |
| class sum (predictions vs. dataset): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 21436 in dataset | 21436 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 in dataset | 46959 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 in dataset | 54828 predicted | pass |
| class sum (predictions vs. dataset): benign_degradation | 270680 in dataset | 270680 predicted | pass |
| class sum (predictions vs. dataset): normal | 20385924 in dataset | 20385924 predicted | pass |
| fold-00: recorded accuracy matches recomputation | 0.986673734316 | 0.986673734316 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.263953121484 | 0.263953121484 | pass |
| fold-01: recorded accuracy matches recomputation | 0.990791397423 | 0.990791397423 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.310963398268 | 0.310963398268 | pass |
| fold-02: recorded accuracy matches recomputation | 0.984641057020 | 0.984641057020 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.276415181400 | 0.276415181400 | pass |
| fold-03: recorded accuracy matches recomputation | 0.973247077664 | 0.973247077664 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.261138749729 | 0.261138749729 | pass |
| fold-04: recorded accuracy matches recomputation | 0.985382277432 | 0.985382277432 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.260013235805 | 0.260013235805 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `grouped-validation-full` | 0.9862 | 0.3014 | 0.2640 | 0.2794 | 0.9789 | 0.9862 | 0.9823 |
| `grouped-validation-full-downsample` | 0.5234 | 0.2263 | 0.6530 | 0.1993 | 0.9826 | 0.5234 | 0.6766 |
| `grouped-validation-full-smote` | 0.9855 | 0.3021 | 0.2625 | 0.2783 | 0.9788 | 0.9855 | 0.9819 |

### Per-class (pooled over folds)

#### `grouped-validation-full`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8209 | 0.5855 | 0.6835 | 270,680 |
| normal | 0.9877 | 0.9983 | 0.9930 | 20,385,924 |

#### `grouped-validation-full-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0150 | 0.9208 | 0.0294 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0050 | 0.5641 | 0.0099 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0088 | 0.6392 | 0.0174 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0123 | 0.6088 | 0.0240 | 54,828 |
| benign_degradation | 0.3183 | 0.6644 | 0.4304 | 270,680 |
| normal | 0.9982 | 0.5207 | 0.6844 | 20,385,924 |

#### `grouped-validation-full-smote`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0018 | 0.0009 | 0.0012 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0009 | 0.0000 | 0.0001 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0069 | 0.0004 | 0.0008 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8150 | 0.5760 | 0.6750 | 270,680 |
| normal | 0.9877 | 0.9977 | 0.9927 | 20,385,924 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `grouped-validation-full` | `grouped-validation-full-downsample` | yes | 20,796,921 | 10,768,091 | 9,741,717 | 117,232 | 169,881 | 9,858,949 |
| `grouped-validation-full` | `grouped-validation-full-smote` | yes | 20,796,921 | 20,489,507 | 20,301 | 6,186 | 280,927 | 26,487 |

