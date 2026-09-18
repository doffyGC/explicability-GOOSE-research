# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-18 17:44:38 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `v2-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 11,057,478 |
| `d2-rule-interval-timestamp` | full_grouped_run | rule:interval-timestamp | none | — | 5 | 11,057,478 |
| `d2-rule-interval-t` | full_grouped_run | rule:interval-t | none | — | 5 | 11,057,478 |
| `d2-rule-time-since-change` | full_grouped_run | rule:time-since-change | none | — | 5 | 11,057,478 |
| `d2-rule-delay` | full_grouped_run | rule:delay | none | — | 5 | 11,057,478 |
| `d2-rule-sqnum-gap` | full_grouped_run | rule:sqnum-gap | none | — | 5 | 11,057,478 |
| `d2-rule-stnum-gap` | full_grouped_run | rule:stnum-gap | none | — | 5 | 11,057,478 |

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

### `d2-rule-interval-timestamp`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 2.98e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.944878857236 | 0.944878857236 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.230452484124 | 0.230452484124 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.921203296087 | 0.921203296087 | pass |
| fold-01: recorded accuracy matches recomputation | 0.964670965207 | 0.964670965207 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.231952480744 | 0.231952480744 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.951107151655 | 0.951107151655 | pass |
| fold-02: recorded accuracy matches recomputation | 0.973049395654 | 0.973049395654 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.226638969283 | 0.226638969283 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.962607122042 | 0.962607122042 | pass |
| fold-03: recorded accuracy matches recomputation | 0.965972132383 | 0.965972132383 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.246231935103 | 0.246231935103 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.951571351173 | 0.951571351173 | pass |
| fold-04: recorded accuracy matches recomputation | 0.939140437036 | 0.939140437036 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.238071175178 | 0.238071175178 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.914146290996 | 0.914146290996 | pass |

</details>

### `d2-rule-interval-t`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 2.98e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.892797707982 | 0.892797707982 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.160900442264 | 0.160900442264 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.894259826209 | 0.894259826209 | pass |
| fold-01: recorded accuracy matches recomputation | 0.911437259008 | 0.911437259008 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.163052821256 | 0.163052821256 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.922076614965 | 0.922076614965 | pass |
| fold-02: recorded accuracy matches recomputation | 0.916748811568 | 0.916748811568 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.161864642037 | 0.161864642037 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.932701618722 | 0.932701618722 | pass |
| fold-03: recorded accuracy matches recomputation | 0.910028469649 | 0.910028469649 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.162386928364 | 0.162386928364 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.921437332608 | 0.921437332608 | pass |
| fold-04: recorded accuracy matches recomputation | 0.885897910134 | 0.885897910134 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.162239036240 | 0.162239036240 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.883533623875 | 0.883533623875 | pass |

</details>

### `d2-rule-time-since-change`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 2.98e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.005182162569 | 0.005182162569 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.001718482905 | 0.001718482905 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.000053432747 | 0.000053432747 | pass |
| fold-01: recorded accuracy matches recomputation | 0.006159625827 | 0.006159625827 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.002040907617 | 0.002040907617 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.000075427364 | 0.000075427364 | pass |
| fold-02: recorded accuracy matches recomputation | 0.003629993254 | 0.003629993254 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.001205703145 | 0.001205703145 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.000026260166 | 0.000026260166 | pass |
| fold-03: recorded accuracy matches recomputation | 0.005813152183 | 0.005813152183 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.001926626767 | 0.001926626767 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.000067198648 | 0.000067198648 | pass |
| fold-04: recorded accuracy matches recomputation | 0.009642465272 | 0.009642465272 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.003185239676 | 0.003185239676 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.000184281378 | 0.000184281378 | pass |

</details>

### `d2-rule-delay`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 2.98e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.272151753348 | 0.272151753348 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.075813963427 | 0.075813963427 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.415324871009 | 0.415324871009 | pass |
| fold-01: recorded accuracy matches recomputation | 0.292268295733 | 0.292268295733 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.079114290834 | 0.079114290834 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.440298685572 | 0.440298685572 | pass |
| fold-02: recorded accuracy matches recomputation | 0.298390789620 | 0.298390789620 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.079129014836 | 0.079129014836 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.451573691691 | 0.451573691691 | pass |
| fold-03: recorded accuracy matches recomputation | 0.290539654999 | 0.290539654999 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.078623861390 | 0.078623861390 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.438724554610 | 0.438724554610 | pass |
| fold-04: recorded accuracy matches recomputation | 0.231533942937 | 0.231533942937 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.067891715433 | 0.067891715433 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.358223597604 | 0.358223597604 | pass |

</details>

### `d2-rule-sqnum-gap`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 2.98e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.907732287028 | 0.907732287028 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.183223638860 | 0.183223638860 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.900860622503 | 0.900860622503 | pass |
| fold-01: recorded accuracy matches recomputation | 0.928119957577 | 0.928119957577 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.189667792875 | 0.189667792875 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.929143076217 | 0.929143076217 | pass |
| fold-02: recorded accuracy matches recomputation | 0.932084682029 | 0.932084682029 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.178530789179 | 0.178530789179 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.940435595441 | 0.940435595441 | pass |
| fold-03: recorded accuracy matches recomputation | 0.927029931604 | 0.927029931604 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.187957466126 | 0.187957466126 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.929351382236 | 0.929351382236 | pass |
| fold-04: recorded accuracy matches recomputation | 0.902827429314 | 0.902827429314 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.195695159498 | 0.195695159498 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.893287238570 | 0.893287238570 | pass |

</details>

### `d2-rule-stnum-gap`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 2.98e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.931124597213 | 0.931124597213 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.163317622159 | 0.163317622159 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.912828857686 | 0.912828857686 | pass |
| fold-01: recorded accuracy matches recomputation | 0.950263022730 | 0.950263022730 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.165451735625 | 0.165451735625 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.941353057389 | 0.941353057389 | pass |
| fold-02: recorded accuracy matches recomputation | 0.959315305014 | 0.959315305014 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.165137889568 | 0.165137889568 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.954098359988 | 0.954098359988 | pass |
| fold-03: recorded accuracy matches recomputation | 0.950771267893 | 0.950771267893 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.165118821053 | 0.165118821053 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.941523262062 | 0.941523262062 | pass |
| fold-04: recorded accuracy matches recomputation | 0.922848425910 | 0.922848425910 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.163866677915 | 0.163866677915 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.901069311088 | 0.901069311088 | pass |

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
| `d2-rule-interval-timestamp` | 0.9597 | 0.2117 | 0.2876 | 0.2350 | 0.9280 | 0.9597 | 0.9431 |
| `d2-rule-interval-t` | 0.9053 | 0.1624 | 0.1745 | 0.1621 | 0.9229 | 0.9053 | 0.9138 |
| `d2-rule-time-since-change` | 0.0058 | 0.0010 | 0.1667 | 0.0019 | 0.0000 | 0.0058 | 0.0001 |
| `d2-rule-delay` | 0.2809 | 0.1663 | 0.2088 | 0.0769 | 0.9466 | 0.2809 | 0.4266 |
| `d2-rule-sqnum-gap` | 0.9214 | 0.1756 | 0.2892 | 0.1866 | 0.9247 | 0.9214 | 0.9215 |
| `d2-rule-stnum-gap` | 0.9451 | 0.1620 | 0.1693 | 0.1647 | 0.9220 | 0.9451 | 0.9334 |

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

#### `d2-rule-interval-timestamp`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.3014 | 0.7261 | 0.4260 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0000 | 0.0000 | 0.0000 | 270,642 |
| normal | 0.9690 | 0.9995 | 0.9840 | 10,570,307 |

#### `d2-rule-interval-t`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0093 | 0.1007 | 0.0170 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0000 | 0.0000 | 0.0000 | 270,642 |
| normal | 0.9654 | 0.9464 | 0.9558 | 10,570,307 |

#### `d2-rule-time-since-change`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0058 | 1.0000 | 0.0115 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0000 | 0.0000 | 0.0000 | 270,642 |
| normal | 0.0000 | 0.0000 | 0.0000 | 10,570,307 |

#### `d2-rule-delay`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0077 | 0.9648 | 0.0153 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0000 | 0.0000 | 0.0000 | 270,642 |
| normal | 0.9902 | 0.2880 | 0.4462 | 10,570,307 |

#### `d2-rule-sqnum-gap`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0870 | 0.7759 | 0.1565 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0000 | 0.0000 | 0.0000 | 270,642 |
| normal | 0.9668 | 0.9592 | 0.9630 | 10,570,307 |

#### `d2-rule-stnum-gap`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0078 | 0.0273 | 0.0121 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0000 | 0.0000 | 0.0000 | 270,642 |
| normal | 0.9645 | 0.9885 | 0.9763 | 10,570,307 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `v2-xgboost-none` | `d2-rule-interval-timestamp` | yes | 11,057,478 | 10,596,463 | 256,650 | 14,908 | 189,457 | 271,558 |
| `v2-xgboost-none` | `d2-rule-interval-t` | yes | 11,057,478 | 9,997,897 | 855,216 | 12,399 | 191,966 | 867,615 |
| `v2-xgboost-none` | `d2-rule-time-since-change` | yes | 11,057,478 | 41,761 | 10,811,352 | 22,202 | 182,163 | 10,833,554 |
| `v2-xgboost-none` | `d2-rule-delay` | yes | 11,057,478 | 3,084,238 | 7,768,875 | 21,626 | 182,739 | 7,790,501 |
| `v2-xgboost-none` | `d2-rule-sqnum-gap` | yes | 11,057,478 | 10,174,200 | 678,913 | 14,458 | 189,907 | 693,371 |
| `v2-xgboost-none` | `d2-rule-stnum-gap` | yes | 11,057,478 | 10,441,058 | 412,055 | 9,280 | 195,085 | 421,335 |

