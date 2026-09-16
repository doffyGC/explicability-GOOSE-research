# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-16 03:54:31 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `v2-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `v2-xgboost-downsample` | full_grouped_run | xgboost | downsample | — | 5 | 11,057,478 |
| `v2-decision-tree-none` | full_grouped_run | decision-tree | none | — | 5 | 11,057,478 |
| `v2-decision-tree-downsample` | full_grouped_run | decision-tree | downsample | — | 5 | 11,057,478 |
| `v2-decision-tree-none-cap4m` | full_grouped_run | decision-tree | none | 4,000,000 | 5 | 11,057,478 |
| `v2-logistic-regression-none` | full_grouped_run | logistic-regression | none | — | 5 | 11,057,478 |
| `v2-logistic-regression-downsample` | full_grouped_run | logistic-regression | downsample | — | 5 | 11,057,478 |
| `v2-random-forest-none-cap4m` | full_grouped_run | random-forest | none | 4,000,000 | 5 | 11,057,478 |
| `v2-random-forest-downsample` | full_grouped_run | random-forest | downsample | — | 5 | 11,057,478 |

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

### `v2-xgboost-downsample`

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
| fold-00: recorded accuracy matches recomputation | 0.789467253356 | 0.789467253356 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.473394028134 | 0.473394028134 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.849632099610 | 0.849632099610 | pass |
| fold-01: recorded accuracy matches recomputation | 0.799717180791 | 0.799717180791 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.467362229408 | 0.467362229408 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.866745147007 | 0.866745147007 | pass |
| fold-02: recorded accuracy matches recomputation | 0.801021317654 | 0.801021317654 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.418960877044 | 0.418960877044 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.870650509369 | 0.870650509369 | pass |
| fold-03: recorded accuracy matches recomputation | 0.791806710787 | 0.791806710787 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.449418354225 | 0.449418354225 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.859434908597 | 0.859434908597 | pass |
| fold-04: recorded accuracy matches recomputation | 0.741534772571 | 0.741534772571 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.469267691793 | 0.469267691793 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.812994084561 | 0.812994084561 | pass |

</details>

### `v2-decision-tree-none`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 4.47e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.975117995246 | 0.975117995246 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.702499183320 | 0.702499183320 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.972050956688 | 0.972050956688 | pass |
| fold-01: recorded accuracy matches recomputation | 0.984382633274 | 0.984382633274 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.710373459930 | 0.710373459930 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982484755959 | 0.982484755959 | pass |
| fold-02: recorded accuracy matches recomputation | 0.987027633264 | 0.987027633264 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.705542122566 | 0.705542122566 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.985463531184 | 0.985463531184 | pass |
| fold-03: recorded accuracy matches recomputation | 0.980226827084 | 0.980226827084 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.694585762223 | 0.694585762223 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.976523129220 | 0.976523129220 | pass |
| fold-04: recorded accuracy matches recomputation | 0.966734804758 | 0.966734804758 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.692724252414 | 0.692724252414 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.962179017627 | 0.962179017627 | pass |

</details>

### `v2-decision-tree-downsample`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 4.47e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.739503835915 | 0.739503835915 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.448187402376 | 0.448187402376 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.816908718218 | 0.816908718218 | pass |
| fold-01: recorded accuracy matches recomputation | 0.780664529945 | 0.780664529945 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.444264270893 | 0.444264270893 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.854464090835 | 0.854464090835 | pass |
| fold-02: recorded accuracy matches recomputation | 0.781623895802 | 0.781623895802 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.400291366583 | 0.400291366583 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.858979957927 | 0.858979957927 | pass |
| fold-03: recorded accuracy matches recomputation | 0.767253224291 | 0.767253224291 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.422580214348 | 0.422580214348 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.843640402462 | 0.843640402462 | pass |
| fold-04: recorded accuracy matches recomputation | 0.714207384739 | 0.714207384739 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.458146880398 | 0.458146880398 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.795188988727 | 0.795188988727 | pass |

</details>

### `v2-decision-tree-none-cap4m`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 3.75e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.975133146340 | 0.975133146340 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.703454652948 | 0.703454652948 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.972086238831 | 0.972086238831 | pass |
| fold-01: recorded accuracy matches recomputation | 0.984359699647 | 0.984359699647 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.709048107116 | 0.709048107116 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982436289738 | 0.982436289738 | pass |
| fold-02: recorded accuracy matches recomputation | 0.986893297525 | 0.986893297525 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.705975807025 | 0.705975807025 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.985380928688 | 0.985380928688 | pass |
| fold-03: recorded accuracy matches recomputation | 0.980135930523 | 0.980135930523 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.691406861841 | 0.691406861841 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.976376800839 | 0.976376800839 | pass |
| fold-04: recorded accuracy matches recomputation | 0.966691763608 | 0.966691763608 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.692882477747 | 0.692882477747 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.962139377103 | 0.962139377103 | pass |

</details>

### `v2-logistic-regression-none`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 4.94e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.948271235845 | 0.948271235845 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.345174694195 | 0.345174694195 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.929748164885 | 0.929748164885 | pass |
| fold-01: recorded accuracy matches recomputation | 0.968287987496 | 0.968287987496 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.374176618855 | 0.374176618855 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.958458299221 | 0.958458299221 | pass |
| fold-02: recorded accuracy matches recomputation | 0.975867883912 | 0.975867883912 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.378388635522 | 0.378388635522 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.968674740613 | 0.968674740613 | pass |
| fold-03: recorded accuracy matches recomputation | 0.965768777750 | 0.965768777750 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.348212559701 | 0.348212559701 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.953808161639 | 0.953808161639 | pass |
| fold-04: recorded accuracy matches recomputation | 0.942849461393 | 0.942849461393 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.389031104780 | 0.389031104780 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.923163391574 | 0.923163391574 | pass |

</details>

### `v2-logistic-regression-downsample`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 5.15e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.687283883098 | 0.687283883098 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.328560174921 | 0.328560174921 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.785683635437 | 0.785683635437 | pass |
| fold-01: recorded accuracy matches recomputation | 0.716574997068 | 0.716574997068 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.321985101659 | 0.321985101659 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.813201764216 | 0.813201764216 | pass |
| fold-02: recorded accuracy matches recomputation | 0.719497459883 | 0.719497459883 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.283671469012 | 0.283671469012 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.818907782718 | 0.818907782718 | pass |
| fold-03: recorded accuracy matches recomputation | 0.701045860063 | 0.701045860063 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.299654972895 | 0.299654972895 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.800876952168 | 0.800876952168 | pass |
| fold-04: recorded accuracy matches recomputation | 0.610315404303 | 0.610315404303 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.319311687193 | 0.319311687193 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.723532677302 | 0.723532677302 | pass |

</details>

### `v2-random-forest-none-cap4m`

39 checks, 1 failed.

| check | expected | observed | result |
|---|---|---|---|
| argmax(posterior) reproduces y_pred on every scored row | 0 mismatches | 7 mismatches | **FAIL** |

### `v2-random-forest-downsample`

39 checks, 1 failed.

| check | expected | observed | result |
|---|---|---|---|
| argmax(posterior) reproduces y_pred on every scored row | 0 mismatches | 17 mismatches | **FAIL** |

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v2-xgboost-none` | 0.9815 | 0.8233 | 0.6852 | 0.7310 | 0.9799 | 0.9815 | 0.9793 |
| `v2-xgboost-downsample` | 0.7880 | 0.3826 | 0.8394 | 0.4594 | 0.9648 | 0.7880 | 0.8548 |
| `v2-decision-tree-none` | 0.9799 | 0.8173 | 0.6529 | 0.7043 | 0.9778 | 0.9799 | 0.9771 |
| `v2-decision-tree-downsample` | 0.7608 | 0.3681 | 0.8145 | 0.4376 | 0.9642 | 0.7608 | 0.8374 |
| `v2-decision-tree-none-cap4m` | 0.9798 | 0.8167 | 0.6505 | 0.7033 | 0.9777 | 0.9798 | 0.9770 |
| `v2-logistic-regression-none` | 0.9622 | 0.4639 | 0.3399 | 0.3677 | 0.9437 | 0.9622 | 0.9496 |
| `v2-logistic-regression-downsample` | 0.6931 | 0.2817 | 0.6555 | 0.3131 | 0.9580 | 0.6931 | 0.7939 |
| `v2-random-forest-none-cap4m` | 0.9815 | 0.7928 | 0.6953 | 0.7355 | 0.9798 | 0.9815 | 0.9799 |
| `v2-random-forest-downsample` | 0.8048 | 0.3757 | 0.8299 | 0.4489 | 0.9642 | 0.8048 | 0.8653 |

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

#### `v2-xgboost-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.5187 | 0.9583 | 0.6731 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.2733 | 0.7780 | 0.4045 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.2962 | 0.8339 | 0.4372 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0547 | 0.8197 | 0.1025 | 54,828 |
| benign_degradation | 0.1528 | 0.8618 | 0.2595 | 270,642 |
| normal | 0.9996 | 0.7850 | 0.8794 | 10,570,307 |

#### `v2-decision-tree-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8146 | 0.8421 | 0.8281 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.7058 | 0.6590 | 0.6816 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8333 | 0.6741 | 0.7453 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6562 | 0.1819 | 0.2848 | 54,828 |
| benign_degradation | 0.9099 | 0.5617 | 0.6946 | 270,642 |
| normal | 0.9843 | 0.9987 | 0.9914 | 10,570,307 |

#### `v2-decision-tree-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.4858 | 0.9585 | 0.6448 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.2645 | 0.7247 | 0.3876 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.2741 | 0.7928 | 0.4074 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0431 | 0.8254 | 0.0820 | 54,828 |
| benign_degradation | 0.1415 | 0.8279 | 0.2417 | 270,642 |
| normal | 0.9997 | 0.7579 | 0.8621 | 10,570,307 |

#### `v2-decision-tree-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8162 | 0.8370 | 0.8265 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.7074 | 0.6532 | 0.6792 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8272 | 0.6663 | 0.7381 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6554 | 0.1865 | 0.2904 | 54,828 |
| benign_degradation | 0.9100 | 0.5615 | 0.6945 | 270,642 |
| normal | 0.9842 | 0.9987 | 0.9914 | 10,570,307 |

#### `v2-logistic-regression-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.5624 | 0.5953 | 0.5784 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.6826 | 0.3032 | 0.4199 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0369 | 0.0007 | 0.0013 | 54,828 |
| benign_degradation | 0.5340 | 0.1413 | 0.2234 | 270,642 |
| normal | 0.9676 | 0.9988 | 0.9829 | 10,570,307 |

#### `v2-logistic-regression-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.4129 | 0.9266 | 0.5713 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0290 | 0.3769 | 0.0538 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.1124 | 0.7057 | 0.1939 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0351 | 0.7529 | 0.0670 | 54,828 |
| benign_degradation | 0.1043 | 0.4718 | 0.1709 | 270,642 |
| normal | 0.9966 | 0.6992 | 0.8219 | 10,570,307 |

#### `v2-random-forest-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.8186 | 0.8415 | 0.8299 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6408 | 0.5954 | 0.6173 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8128 | 0.7296 | 0.7690 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.6151 | 0.3965 | 0.4822 | 54,828 |
| benign_degradation | 0.8818 | 0.6110 | 0.7218 | 270,642 |
| normal | 0.9878 | 0.9981 | 0.9929 | 10,570,307 |

#### `v2-random-forest-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.5716 | 0.8962 | 0.6980 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.2161 | 0.7814 | 0.3385 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.2381 | 0.8345 | 0.3705 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0646 | 0.8161 | 0.1196 | 54,828 |
| benign_degradation | 0.1651 | 0.8479 | 0.2764 | 270,642 |
| normal | 0.9989 | 0.8032 | 0.8904 | 10,570,307 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `v2-xgboost-none` | `v2-xgboost-downsample` | yes | 11,057,478 | 8,582,929 | 2,270,184 | 130,533 | 73,832 | 2,400,717 |
| `v2-xgboost-none` | `v2-decision-tree-none` | yes | 11,057,478 | 10,816,357 | 36,756 | 18,581 | 185,784 | 55,337 |
| `v2-xgboost-none` | `v2-decision-tree-downsample` | yes | 11,057,478 | 8,282,934 | 2,570,179 | 129,390 | 74,975 | 2,699,569 |
| `v2-xgboost-none` | `v2-decision-tree-none-cap4m` | yes | 11,057,478 | 10,815,484 | 37,629 | 18,781 | 185,584 | 56,410 |
| `v2-xgboost-none` | `v2-logistic-regression-none` | yes | 11,057,478 | 10,619,394 | 233,719 | 20,442 | 183,923 | 254,161 |
| `v2-xgboost-none` | `v2-logistic-regression-downsample` | yes | 11,057,478 | 7,560,483 | 3,292,630 | 104,004 | 100,361 | 3,396,634 |
| `v2-xgboost-none` | `v2-random-forest-none-cap4m` | yes | 11,057,478 | 10,815,569 | 37,544 | 36,933 | 167,432 | 74,477 |
| `v2-xgboost-none` | `v2-random-forest-downsample` | yes | 11,057,478 | 8,768,065 | 2,085,048 | 130,651 | 73,714 | 2,215,699 |

