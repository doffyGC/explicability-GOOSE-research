# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-13 14:15:00 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `d5-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 23,226,530 |
| `d5-xgboost-downsample` | full_grouped_run | xgboost | downsample | — | 5 | 23,226,530 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `d5-xgboost-none`

39 checks, 0 failed.

<details><summary>All checks passed - expand for detail</summary>

| check | expected | observed | result |
|---|---|---|---|
| each row_index predicted at most once | 0 duplicates | 0 duplicates | pass |
| prediction rows == report rows_used | 23226530 | 23226530 | pass |
| full run covers dataset rows 0..N-1 with no gaps | 0..23226529 | 0..23226529 over 23226530 rows | pass |
| fold-00: predictions == test_rows == sum(per-class support) | 6170527 == 6170527 | 6170527 predictions | pass |
| fold-01: predictions == test_rows == sum(per-class support) | 5413230 == 5413230 | 5413230 predictions | pass |
| fold-02: predictions == test_rows == sum(per-class support) | 3523612 == 3523612 | 3523612 predictions | pass |
| fold-03: predictions == test_rows == sum(per-class support) | 4116401 == 4116401 | 4116401 predictions | pass |
| fold-04: predictions == test_rows == sum(per-class support) | 4002760 == 4002760 | 4002760 predictions | pass |
| scored rows == predicted rows | 23226530 | 23226530 scored, 0 not found in predictions | pass |
| argmax(posterior) reproduces y_pred on every scored row | 0 mismatches | 0 mismatches | pass |
| scores y_true agrees with predictions y_true | 0 mismatches | 0 mismatches | pass |
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 8.92e-08, 0 out of range | pass |
| class sum (predictions vs. report support): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 50782 | 50782 | pass |
| class sum (predictions vs. report support): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 63976 | 63976 | pass |
| class sum (predictions vs. report support): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 | 46959 | pass |
| class sum (predictions vs. report support): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 | 54828 | pass |
| class sum (predictions vs. report support): benign_degradation | 270680 | 270680 | pass |
| class sum (predictions vs. report support): normal | 22739305 | 22739305 | pass |
| class sum (predictions vs. dataset): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 50782 in dataset | 50782 predicted | pass |
| class sum (predictions vs. dataset): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 63976 in dataset | 63976 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 in dataset | 46959 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 in dataset | 54828 predicted | pass |
| class sum (predictions vs. dataset): benign_degradation | 270680 in dataset | 270680 predicted | pass |
| class sum (predictions vs. dataset): normal | 22739305 in dataset | 22739305 predicted | pass |
| fold-00: recorded accuracy matches recomputation | 0.988196794212 | 0.988196794212 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.298063379656 | 0.298063379656 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.984487246726 | 0.984487246726 | pass |
| fold-01: recorded accuracy matches recomputation | 0.987223894052 | 0.987223894052 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.284247384393 | 0.284247384393 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982932311475 | 0.982932311475 | pass |
| fold-02: recorded accuracy matches recomputation | 0.977943087945 | 0.977943087945 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.264093392879 | 0.264093392879 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.971078414382 | 0.971078414382 | pass |
| fold-03: recorded accuracy matches recomputation | 0.982087265065 | 0.982087265065 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.284596926965 | 0.284596926965 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.976712792315 | 0.976712792315 | pass |
| fold-04: recorded accuracy matches recomputation | 0.984601874706 | 0.984601874706 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.220348534164 | 0.220348534164 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.977459112528 | 0.977459112528 | pass |

</details>

### `d5-xgboost-downsample`

39 checks, 0 failed.

<details><summary>All checks passed - expand for detail</summary>

| check | expected | observed | result |
|---|---|---|---|
| each row_index predicted at most once | 0 duplicates | 0 duplicates | pass |
| prediction rows == report rows_used | 23226530 | 23226530 | pass |
| full run covers dataset rows 0..N-1 with no gaps | 0..23226529 | 0..23226529 over 23226530 rows | pass |
| fold-00: predictions == test_rows == sum(per-class support) | 6170527 == 6170527 | 6170527 predictions | pass |
| fold-01: predictions == test_rows == sum(per-class support) | 5413230 == 5413230 | 5413230 predictions | pass |
| fold-02: predictions == test_rows == sum(per-class support) | 3523612 == 3523612 | 3523612 predictions | pass |
| fold-03: predictions == test_rows == sum(per-class support) | 4116401 == 4116401 | 4116401 predictions | pass |
| fold-04: predictions == test_rows == sum(per-class support) | 4002760 == 4002760 | 4002760 predictions | pass |
| scored rows == predicted rows | 23226530 | 23226530 scored, 0 not found in predictions | pass |
| argmax(posterior) reproduces y_pred on every scored row | 0 mismatches | 0 mismatches | pass |
| scores y_true agrees with predictions y_true | 0 mismatches | 0 mismatches | pass |
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 9.71e-08, 0 out of range | pass |
| class sum (predictions vs. report support): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 50782 | 50782 | pass |
| class sum (predictions vs. report support): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 63976 | 63976 | pass |
| class sum (predictions vs. report support): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 | 46959 | pass |
| class sum (predictions vs. report support): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 | 54828 | pass |
| class sum (predictions vs. report support): benign_degradation | 270680 | 270680 | pass |
| class sum (predictions vs. report support): normal | 22739305 | 22739305 | pass |
| class sum (predictions vs. dataset): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 50782 in dataset | 50782 predicted | pass |
| class sum (predictions vs. dataset): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 63976 in dataset | 63976 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 in dataset | 46959 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 in dataset | 54828 predicted | pass |
| class sum (predictions vs. dataset): benign_degradation | 270680 in dataset | 270680 predicted | pass |
| class sum (predictions vs. dataset): normal | 22739305 in dataset | 22739305 predicted | pass |
| fold-00: recorded accuracy matches recomputation | 0.622006353752 | 0.622006353752 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.224066570305 | 0.224066570305 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.755511141086 | 0.755511141086 | pass |
| fold-01: recorded accuracy matches recomputation | 0.581238003927 | 0.581238003927 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.217696578180 | 0.217696578180 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.724442757314 | 0.724442757314 | pass |
| fold-02: recorded accuracy matches recomputation | 0.520959742446 | 0.520959742446 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.194577665939 | 0.194577665939 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.670262045697 | 0.670262045697 | pass |
| fold-03: recorded accuracy matches recomputation | 0.529647378863 | 0.529647378863 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.218236474593 | 0.218236474593 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.677193690091 | 0.677193690091 | pass |
| fold-04: recorded accuracy matches recomputation | 0.524808632044 | 0.524808632044 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.158131061569 | 0.158131061569 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.676728546992 | 0.676728546992 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d5-xgboost-none` | 0.9847 | 0.3056 | 0.2665 | 0.2823 | 0.9751 | 0.9847 | 0.9796 |
| `d5-xgboost-downsample` | 0.5641 | 0.2253 | 0.6721 | 0.2099 | 0.9806 | 0.5641 | 0.7085 |

### Per-class (pooled over folds)

#### `d5-xgboost-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0039 | 0.0000 | 0.0000 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8438 | 0.6003 | 0.7015 | 270,680 |
| normal | 0.9859 | 0.9987 | 0.9922 | 22,739,305 |

#### `d5-xgboost-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0370 | 0.9024 | 0.0712 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0126 | 0.6292 | 0.0248 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0124 | 0.6765 | 0.0243 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0112 | 0.5716 | 0.0221 | 54,828 |
| benign_degradation | 0.2801 | 0.6916 | 0.3988 | 270,680 |
| normal | 0.9981 | 0.5613 | 0.7186 | 22,739,305 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `d5-xgboost-none` | `d5-xgboost-downsample` | yes | 23,226,530 | 12,926,977 | 9,944,470 | 174,089 | 180,994 | 10,118,559 |

