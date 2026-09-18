# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-18 15:45:02 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `v2-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-timing-deltas` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-size-state-deltas` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-delta` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-sequence` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `v2-xgboost-none`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 8.93e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.977025566736 | 0.977025566736 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.733743369866 | 0.733743369866 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.974375169205 | 0.974375169205 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985510275346 | 0.985510275346 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.735718323159 | 0.735718323159 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.984024919576 | 0.984024919576 | pass |
| fold-02: recorded accuracy matches recomputation | 0.988549982961 | 0.988549982961 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.742641940577 | 0.742641940577 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.987471735223 | 0.987471735223 | pass |
| fold-03: recorded accuracy matches recomputation | 0.982110711166 | 0.982110711166 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.726151396718 | 0.726151396718 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.979176963669 | 0.979176963669 | pass |
| fold-04: recorded accuracy matches recomputation | 0.968637972151 | 0.968637972151 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.705797669577 | 0.705797669577 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.964729536554 | 0.964729536554 | pass |

</details>

### `d1-xgboost-no-timing-deltas`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 8.93e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.965466748971 | 0.965466748971 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.644928235783 | 0.644928235783 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.959199612998 | 0.959199612998 | pass |
| fold-01: recorded accuracy matches recomputation | 0.979487818566 | 0.979487818566 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.649208311406 | 0.649208311406 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.976112675264 | 0.976112675264 | pass |
| fold-02: recorded accuracy matches recomputation | 0.983828466073 | 0.983828466073 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.648371780329 | 0.648371780329 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.981989568958 | 0.981989568958 | pass |
| fold-03: recorded accuracy matches recomputation | 0.978295592235 | 0.978295592235 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.663847352385 | 0.663847352385 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.974064305003 | 0.974064305003 | pass |
| fold-04: recorded accuracy matches recomputation | 0.956999395553 | 0.956999395553 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.615181800566 | 0.615181800566 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.949700223597 | 0.949700223597 | pass |

</details>

### `d1-xgboost-no-size-state-deltas`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 9.35e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.976991354591 | 0.976991354591 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.734279683109 | 0.734279683109 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.974344236220 | 0.974344236220 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985519794965 | 0.985519794965 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.734418855233 | 0.734418855233 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.984023102297 | 0.984023102297 | pass |
| fold-02: recorded accuracy matches recomputation | 0.988445296254 | 0.988445296254 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.742965912278 | 0.742965912278 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.987389534049 | 0.987389534049 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981934414115 | 0.981934414115 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.720985857599 | 0.720985857599 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.978877914877 | 0.978877914877 | pass |
| fold-04: recorded accuracy matches recomputation | 0.968578712596 | 0.968578712596 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.706367239981 | 0.706367239981 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.964650260907 | 0.964650260907 | pass |

</details>

### `d1-xgboost-no-delta`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 8.92e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.951612784981 | 0.951612784981 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.241321671946 | 0.241321671946 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.934696760057 | 0.934696760057 | pass |
| fold-01: recorded accuracy matches recomputation | 0.964749718414 | 0.964749718414 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.221356593833 | 0.221356593833 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.950112240089 | 0.950112240089 | pass |
| fold-02: recorded accuracy matches recomputation | 0.973184463468 | 0.973184463468 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.220614847620 | 0.220614847620 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.962399997841 | 0.962399997841 | pass |
| fold-03: recorded accuracy matches recomputation | 0.965784843189 | 0.965784843189 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.225779308180 | 0.225779308180 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.951404597376 | 0.951404597376 | pass |
| fold-04: recorded accuracy matches recomputation | 0.941543256044 | 0.941543256044 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.220555711008 | 0.220555711008 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.918941143370 | 0.918941143370 | pass |

</details>

### `d1-xgboost-no-sequence`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 9.04e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.975709376620 | 0.975709376620 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.714991365560 | 0.714991365560 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.972742964156 | 0.972742964156 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985356230612 | 0.985356230612 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.731918732456 | 0.731918732456 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.983886388077 | 0.983886388077 | pass |
| fold-02: recorded accuracy matches recomputation | 0.987365119780 | 0.987365119780 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.718976835154 | 0.718976835154 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.985937556180 | 0.985937556180 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981018684105 | 0.981018684105 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.700278865240 | 0.700278865240 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.977793511484 | 0.977793511484 | pass |
| fold-04: recorded accuracy matches recomputation | 0.967358589548 | 0.967358589548 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.696879774881 | 0.696879774881 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.962537196832 | 0.962537196832 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v2-xgboost-none` | 0.9815 | 0.8233 | 0.6852 | 0.7310 | 0.9799 | 0.9815 | 0.9793 |
| `d1-xgboost-no-timing-deltas` | 0.9745 | 0.7548 | 0.5972 | 0.6450 | 0.9714 | 0.9745 | 0.9702 |
| `d1-xgboost-no-size-state-deltas` | 0.9814 | 0.8216 | 0.6847 | 0.7296 | 0.9798 | 0.9814 | 0.9792 |
| `d1-xgboost-no-delta` | 0.9613 | 0.3551 | 0.2058 | 0.2275 | 0.9439 | 0.9613 | 0.9460 |
| `d1-xgboost-no-sequence` | 0.9805 | 0.8122 | 0.6674 | 0.7155 | 0.9788 | 0.9805 | 0.9779 |

### Per-class (pooled over folds)

#### `v2-xgboost-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8298 | 0.8828 | 0.8555 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6634 | 0.6529 | 0.6581 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8407 | 0.7272 | 0.7798 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.7064 | 0.2721 | 0.3929 | 54,828 |
| benign_degradation | 0.9134 | 0.5768 | 0.7071 | 270,642 |
| normal | 0.9863 | 0.9992 | 0.9927 | 10,570,307 |

#### `d1-xgboost-no-timing-deltas`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.7856 | 0.8241 | 0.8044 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.5095 | 0.5187 | 0.5141 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8124 | 0.6441 | 0.7185 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.5931 | 0.2208 | 0.3218 | 54,828 |
| benign_degradation | 0.8470 | 0.3767 | 0.5215 | 270,642 |
| normal | 0.9809 | 0.9986 | 0.9897 | 10,570,307 |

#### `d1-xgboost-no-size-state-deltas`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8295 | 0.8832 | 0.8555 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6616 | 0.6575 | 0.6596 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8403 | 0.7285 | 0.7804 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6999 | 0.2636 | 0.3830 | 54,828 |
| benign_degradation | 0.9122 | 0.5763 | 0.7063 | 270,642 |
| normal | 0.9863 | 0.9991 | 0.9927 | 10,570,307 |

#### `d1-xgboost-no-delta`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.1834 | 0.0090 | 0.0172 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.9859 | 0.2261 | 0.3679 | 270,642 |
| normal | 0.9613 | 0.9997 | 0.9801 | 10,570,307 |

#### `d1-xgboost-no-sequence`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8147 | 0.8685 | 0.8408 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6377 | 0.6176 | 0.6275 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8333 | 0.7110 | 0.7673 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6881 | 0.2710 | 0.3888 | 54,828 |
| benign_degradation | 0.9140 | 0.5365 | 0.6761 | 270,642 |
| normal | 0.9855 | 0.9995 | 0.9924 | 10,570,307 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `v2-xgboost-none` | `d1-xgboost-no-timing-deltas` | yes | 11,057,478 | 10,761,656 | 91,457 | 13,308 | 191,057 | 104,765 |
| `v2-xgboost-none` | `d1-xgboost-no-size-state-deltas` | yes | 11,057,478 | 10,846,554 | 6,559 | 5,713 | 198,652 | 12,272 |
| `v2-xgboost-none` | `d1-xgboost-no-delta` | yes | 11,057,478 | 10,619,129 | 233,984 | 9,980 | 194,385 | 243,964 |
| `v2-xgboost-none` | `d1-xgboost-no-sequence` | yes | 11,057,478 | 10,828,653 | 24,460 | 13,540 | 190,825 | 38,000 |

