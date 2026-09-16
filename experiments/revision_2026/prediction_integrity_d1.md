# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-16 18:37:23 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `v2-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-electrical` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-goose-header` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-absolute-time` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-delta` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d1-xgboost-no-counters` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
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

### `d1-xgboost-no-electrical`

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
| fold-00: recorded accuracy matches recomputation | 0.976232333703 | 0.976232333703 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.732348612214 | 0.732348612214 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.973386440401 | 0.973386440401 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985297814772 | 0.985297814772 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.733753050894 | 0.733753050894 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.983767985631 | 0.983767985631 | pass |
| fold-02: recorded accuracy matches recomputation | 0.988351224633 | 0.988351224633 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.742762299746 | 0.742762299746 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.987260530924 | 0.987260530924 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981703579126 | 0.981703579126 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.721000754687 | 0.721000754687 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.978654798209 | 0.978654798209 | pass |
| fold-04: recorded accuracy matches recomputation | 0.968307366212 | 0.968307366212 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.705721322491 | 0.705721322491 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.964269163097 | 0.964269163097 | pass |

</details>

### `d1-xgboost-no-goose-header`

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
| fold-00: recorded accuracy matches recomputation | 0.976938081393 | 0.976938081393 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.735706975298 | 0.735706975298 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.974293263600 | 0.974293263600 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985506380957 | 0.985506380957 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.735385607783 | 0.735385607783 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.984019173426 | 0.984019173426 | pass |
| fold-02: recorded accuracy matches recomputation | 0.988517405629 | 0.988517405629 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.742218370514 | 0.742218370514 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.987450549085 | 0.987450549085 | pass |
| fold-03: recorded accuracy matches recomputation | 0.982020237380 | 0.982020237380 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.723138406102 | 0.723138406102 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.979032807659 | 0.979032807659 | pass |
| fold-04: recorded accuracy matches recomputation | 0.968650447846 | 0.968650447846 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.706022590141 | 0.706022590141 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.964711112298 | 0.964711112298 | pass |

</details>

### `d1-xgboost-no-absolute-time`

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
| fold-00: recorded accuracy matches recomputation | 0.976844242365 | 0.976844242365 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.733776397994 | 0.733776397994 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.974166442431 | 0.974166442431 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985589461263 | 0.985589461263 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.735277140146 | 0.735277140146 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.984080557768 | 0.984080557768 | pass |
| fold-02: recorded accuracy matches recomputation | 0.988314620890 | 0.988314620890 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.735150405047 | 0.735150405047 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.987229551833 | 0.987229551833 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981810963901 | 0.981810963901 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.715833413617 | 0.715833413617 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.978709512453 | 0.978709512453 | pass |
| fold-04: recorded accuracy matches recomputation | 0.968957973747 | 0.968957973747 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.708323144569 | 0.708323144569 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.965121544525 | 0.965121544525 | pass |

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

### `d1-xgboost-no-counters`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 8.94e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.976310044148 | 0.976310044148 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.729228966170 | 0.729228966170 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.973611637208 | 0.973611637208 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985649607943 | 0.985649607943 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.738037149339 | 0.738037149339 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.984168984654 | 0.984168984654 | pass |
| fold-02: recorded accuracy matches recomputation | 0.987766296810 | 0.987766296810 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.729694902673 | 0.729694902673 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.986623902027 | 0.986623902027 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981612682565 | 0.981612682565 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.716365631600 | 0.716365631600 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.978561772119 | 0.978561772119 | pass |
| fold-04: recorded accuracy matches recomputation | 0.967903777454 | 0.967903777454 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.705095448905 | 0.705095448905 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.963794563492 | 0.963794563492 | pass |

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
| `d1-xgboost-no-electrical` | 0.9811 | 0.8209 | 0.6834 | 0.7288 | 0.9795 | 0.9811 | 0.9788 |
| `d1-xgboost-no-goose-header` | 0.9815 | 0.8222 | 0.6849 | 0.7304 | 0.9799 | 0.9815 | 0.9792 |
| `d1-xgboost-no-absolute-time` | 0.9814 | 0.8202 | 0.6848 | 0.7279 | 0.9799 | 0.9814 | 0.9791 |
| `d1-xgboost-no-delta` | 0.9613 | 0.3551 | 0.2058 | 0.2275 | 0.9439 | 0.9613 | 0.9460 |
| `d1-xgboost-no-counters` | 0.9810 | 0.8173 | 0.6832 | 0.7263 | 0.9794 | 0.9810 | 0.9787 |
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

#### `d1-xgboost-no-electrical`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8295 | 0.8823 | 0.8551 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6633 | 0.6558 | 0.6595 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8371 | 0.7281 | 0.7788 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6997 | 0.2720 | 0.3917 | 54,828 |
| benign_degradation | 0.9098 | 0.5628 | 0.6954 | 270,642 |
| normal | 0.9860 | 0.9991 | 0.9925 | 10,570,307 |

#### `d1-xgboost-no-goose-header`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8290 | 0.8826 | 0.8550 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6608 | 0.6551 | 0.6579 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8408 | 0.7260 | 0.7792 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.7013 | 0.2721 | 0.3921 | 54,828 |
| benign_degradation | 0.9149 | 0.5745 | 0.7058 | 270,642 |
| normal | 0.9863 | 0.9992 | 0.9927 | 10,570,307 |

#### `d1-xgboost-no-absolute-time`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8259 | 0.8871 | 0.8554 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6504 | 0.6719 | 0.6610 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8446 | 0.7212 | 0.7780 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6956 | 0.2560 | 0.3743 | 54,828 |
| benign_degradation | 0.9185 | 0.5737 | 0.7063 | 270,642 |
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

#### `d1-xgboost-no-counters`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8228 | 0.8826 | 0.8516 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6491 | 0.6732 | 0.6609 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8408 | 0.7181 | 0.7746 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6955 | 0.2652 | 0.3839 | 54,828 |
| benign_degradation | 0.9096 | 0.5613 | 0.6942 | 270,642 |
| normal | 0.9861 | 0.9990 | 0.9925 | 10,570,307 |

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
| `v2-xgboost-none` | `d1-xgboost-no-electrical` | yes | 11,057,478 | 10,841,514 | 11,599 | 7,449 | 196,916 | 19,048 |
| `v2-xgboost-none` | `d1-xgboost-no-goose-header` | yes | 11,057,478 | 10,847,308 | 5,805 | 5,334 | 199,031 | 11,139 |
| `v2-xgboost-none` | `d1-xgboost-no-absolute-time` | yes | 11,057,478 | 10,843,772 | 9,341 | 8,314 | 196,051 | 17,655 |
| `v2-xgboost-none` | `d1-xgboost-no-delta` | yes | 11,057,478 | 10,619,129 | 233,984 | 9,980 | 194,385 | 243,964 |
| `v2-xgboost-none` | `d1-xgboost-no-counters` | yes | 11,057,478 | 10,839,098 | 14,015 | 8,377 | 195,988 | 22,392 |
| `v2-xgboost-none` | `d1-xgboost-no-sequence` | yes | 11,057,478 | 10,828,653 | 24,460 | 13,540 | 190,825 | 38,000 |

