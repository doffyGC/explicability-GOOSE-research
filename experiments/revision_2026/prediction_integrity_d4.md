# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-21 11:45:06 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `d4-xgboost-tuned` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `d4-xgboost-tuned`

39 checks, 0 failed.

<details><summary>All checks passed - expand for detail</summary>

| check | expected | observed | result |
|---|---|---|---|
| each row_index predicted at most once | 0 duplicates | 0 duplicates | pass |
| prediction rows == report rows_used | 11057478 | 11057478 | pass |
| full run covers dataset rows 0..N-1 with no gaps | 0..11057477 | 0..11057477 over 11057478 rows | pass |
| fold-00: predictions == test_rows == sum(per-class support) | 2046057 == 2046057 | 2046057 predictions | pass |
| fold-01: predictions == test_rows == sum(per-class support) | 2311017 == 2311017 | 2311017 predictions | pass |
| fold-02: predictions == test_rows == sum(per-class support) | 2731961 == 2731961 | 2731961 predictions | pass |
| fold-03: predictions == test_rows == sum(per-class support) | 2365326 == 2365326 | 2365326 predictions | pass |
| fold-04: predictions == test_rows == sum(per-class support) | 1603117 == 1603117 | 1603117 predictions | pass |
| scored rows == predicted rows | 11057478 | 11057478 scored, 0 not found in predictions | pass |
| argmax(posterior) reproduces y_pred on every scored row | 0 mismatches | 0 mismatches | pass |
| scores y_true agrees with predictions y_true | 0 mismatches | 0 mismatches | pass |
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 9.06e-08, 0 out of range | pass |
| class sum (predictions vs. report support): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 50779 | 50779 | pass |
| class sum (predictions vs. report support): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 63963 | 63963 | pass |
| class sum (predictions vs. report support): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 | 46959 | pass |
| class sum (predictions vs. report support): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 | 54828 | pass |
| class sum (predictions vs. report support): benign_degradation | 270642 | 270642 | pass |
| class sum (predictions vs. report support): normal | 10570307 | 10570307 | pass |
| class sum (predictions vs. dataset): DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 50779 in dataset | 50779 predicted | pass |
| class sum (predictions vs. dataset): FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 63963 in dataset | 63963 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_BURST_ORIENTEDGRAYHOLE | 46959 in dataset | 46959 predicted | pass |
| class sum (predictions vs. dataset): RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 54828 in dataset | 54828 predicted | pass |
| class sum (predictions vs. dataset): benign_degradation | 270642 in dataset | 270642 predicted | pass |
| class sum (predictions vs. dataset): normal | 10570307 in dataset | 10570307 predicted | pass |
| fold-00: recorded accuracy matches recomputation | 0.976885296939 | 0.976885296939 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.733362169772 | 0.733362169772 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.974209470229 | 0.974209470229 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985487341720 | 0.985487341720 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.734635069006 | 0.734635069006 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.983971238261 | 0.983971238261 | pass |
| fold-02: recorded accuracy matches recomputation | 0.988423700045 | 0.988423700045 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.741304175273 | 0.741304175273 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.987339152933 | 0.987339152933 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981909893182 | 0.981909893182 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.722109986578 | 0.722109986578 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.978874748508 | 0.978874748508 | pass |
| fold-04: recorded accuracy matches recomputation | 0.968531928736 | 0.968531928736 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.706207328207 | 0.706207328207 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.964634632364 | 0.964634632364 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d4-xgboost-tuned` | 0.9814 | 0.8222 | 0.6839 | 0.7295 | 0.9798 | 0.9814 | 0.9791 |

### Per-class (pooled over folds)

#### `d4-xgboost-tuned`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8299 | 0.8826 | 0.8554 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6630 | 0.6539 | 0.6584 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8401 | 0.7263 | 0.7791 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.7022 | 0.2668 | 0.3866 | 54,828 |
| benign_degradation | 0.9117 | 0.5745 | 0.7048 | 270,642 |
| normal | 0.9862 | 0.9991 | 0.9926 | 10,570,307 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| - | - | n/a (single run audited) | - | - | - | - | - | - |

