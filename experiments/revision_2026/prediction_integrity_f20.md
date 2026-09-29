# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-29 03:45:35 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `f20-decision-tree-none` | full_grouped_run | decision-tree | none | — | 5 | 11,057,478 |
| `f20-logistic-regression-none` | full_grouped_run | logistic-regression | none | — | 5 | 11,057,478 |
| `f20-random-forest-none-cap4m` | full_grouped_run | random-forest | none | 4,000,000 | 5 | 11,057,478 |
| `f20-xgboost-downsample` | full_grouped_run | xgboost | downsample | — | 5 | 11,057,478 |
| `f20-xgboost-smote` | full_grouped_run | xgboost | smote | — | 5 | 11,057,478 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `f20-decision-tree-none`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 4.56e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.967941264588 | 0.967941264588 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.630500273641 | 0.630500273641 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.961410552977 | 0.961410552977 | pass |
| fold-01: recorded accuracy matches recomputation | 0.980145970367 | 0.980145970367 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.624789294151 | 0.624789294151 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.976353186867 | 0.976353186867 | pass |
| fold-02: recorded accuracy matches recomputation | 0.983712432205 | 0.983712432205 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.626905524429 | 0.626905524429 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.980499769557 | 0.980499769557 | pass |
| fold-03: recorded accuracy matches recomputation | 0.976077293363 | 0.976077293363 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.596902143032 | 0.596902143032 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.969434106460 | 0.969434106460 | pass |
| fold-04: recorded accuracy matches recomputation | 0.961763863773 | 0.961763863773 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.628654940111 | 0.628654940111 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.953549029165 | 0.953549029165 | pass |

</details>

### `f20-logistic-regression-none`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 4.97e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.947302054635 | 0.947302054635 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.295544183461 | 0.295544183461 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.927211050577 | 0.927211050577 | pass |
| fold-01: recorded accuracy matches recomputation | 0.966853554085 | 0.966853554085 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.328540939823 | 0.328540939823 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.955929987407 | 0.955929987407 | pass |
| fold-02: recorded accuracy matches recomputation | 0.974859450775 | 0.974859450775 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.328370859151 | 0.328370859151 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.966409938982 | 0.966409938982 | pass |
| fold-03: recorded accuracy matches recomputation | 0.965091492674 | 0.965091492674 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.303902297325 | 0.303902297325 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.951485478792 | 0.951485478792 | pass |
| fold-04: recorded accuracy matches recomputation | 0.941414756378 | 0.941414756378 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.344407094151 | 0.344407094151 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.920067342040 | 0.920067342040 | pass |

</details>

### `f20-random-forest-none-cap4m`

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
| argmax(posterior) reproduces y_pred on every scored row | 0 mismatches outside exact top-1 ties in a fallback fold | 3 mismatches, 3 of them exact top-1 ties in a fold that fell back to model.predict | pass |
| scores y_true agrees with predictions y_true | 0 mismatches | 0 mismatches | pass |
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 4.84e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.976129208522 | 0.976129208522 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.698396009576 | 0.698396009576 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.973893462140 | 0.973893462140 | pass |
| fold-01: recorded accuracy matches recomputation | 0.984101804530 | 0.984101804530 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.694817231681 | 0.694817231681 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982829738447 | 0.982829738447 | pass |
| fold-02: recorded accuracy matches recomputation | 0.986992127633 | 0.986992127633 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.694932395655 | 0.694932395655 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.985937176019 | 0.985937176019 | pass |
| fold-03: recorded accuracy matches recomputation | 0.980431872816 | 0.980431872816 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.678544499558 | 0.678544499558 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.977496753873 | 0.977496753873 | pass |
| fold-04: recorded accuracy matches recomputation | 0.967224475818 | 0.967224475818 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.671703942729 | 0.671703942729 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.963382645353 | 0.963382645353 | pass |

</details>

### `f20-xgboost-downsample`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 9.26e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.766663880821 | 0.766663880821 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.458853718163 | 0.458853718163 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.833736513384 | 0.833736513384 | pass |
| fold-01: recorded accuracy matches recomputation | 0.771040195723 | 0.771040195723 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.452746665139 | 0.452746665139 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.848139669815 | 0.848139669815 | pass |
| fold-02: recorded accuracy matches recomputation | 0.782489208301 | 0.782489208301 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.398125357609 | 0.398125357609 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.858906510884 | 0.858906510884 | pass |
| fold-03: recorded accuracy matches recomputation | 0.787381950733 | 0.787381950733 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.433975015776 | 0.433975015776 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.856648743231 | 0.856648743231 | pass |
| fold-04: recorded accuracy matches recomputation | 0.724196674354 | 0.724196674354 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.459233752926 | 0.459233752926 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.800913360569 | 0.800913360569 | pass |

</details>

### `f20-xgboost-smote`

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
| posteriors are a distribution (in [0,1], summing to 1) | max |sum - 1| <= 0.0001, 0 values outside [0,1] | max drift 9.13e-08, 0 out of range | pass |
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
| fold-00: recorded accuracy matches recomputation | 0.975339885448 | 0.975339885448 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.724121549290 | 0.724121549290 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.972939215105 | 0.972939215105 | pass |
| fold-01: recorded accuracy matches recomputation | 0.984903616027 | 0.984903616027 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.727943951253 | 0.727943951253 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.983705931859 | 0.983705931859 | pass |
| fold-02: recorded accuracy matches recomputation | 0.987120972810 | 0.987120972810 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.714817493573 | 0.714817493573 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.986181136901 | 0.986181136901 | pass |
| fold-03: recorded accuracy matches recomputation | 0.980657634508 | 0.980657634508 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.713327003333 | 0.713327003333 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.977995261704 | 0.977995261704 | pass |
| fold-04: recorded accuracy matches recomputation | 0.965919518039 | 0.965919518039 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.690612806238 | 0.690612806238 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.962203134781 | 0.962203134781 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `f20-decision-tree-none` | 0.9752 | 0.7984 | 0.5765 | 0.6240 | 0.9725 | 0.9752 | 0.9698 |
| `f20-logistic-regression-none` | 0.9611 | 0.4220 | 0.2920 | 0.3205 | 0.9419 | 0.9611 | 0.9470 |
| `f20-random-forest-none-cap4m` | 0.9801 | 0.7669 | 0.6470 | 0.6913 | 0.9778 | 0.9801 | 0.9780 |
| `f20-xgboost-downsample` | 0.7698 | 0.3723 | 0.8076 | 0.4440 | 0.9619 | 0.7698 | 0.8427 |
| `f20-xgboost-smote` | 0.9800 | 0.7720 | 0.6883 | 0.7181 | 0.9786 | 0.9800 | 0.9780 |

### Per-class (pooled over folds)

#### `f20-decision-tree-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.7814 | 0.8656 | 0.8213 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6913 | 0.6671 | 0.6789 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.8184 | 0.4801 | 0.6051 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.5989 | 0.0535 | 0.0983 | 54,828 |
| benign_degradation | 0.9215 | 0.3931 | 0.5511 | 270,642 |
| normal | 0.9790 | 0.9995 | 0.9892 | 10,570,307 |

#### `f20-logistic-regression-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.5986 | 0.4849 | 0.5358 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.3442 | 0.1522 | 0.2111 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0301 | 0.0009 | 0.0018 | 54,828 |
| benign_degradation | 0.5938 | 0.1145 | 0.1920 | 270,642 |
| normal | 0.9656 | 0.9995 | 0.9823 | 10,570,307 |

#### `f20-random-forest-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.7878 | 0.7707 | 0.7792 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6434 | 0.5498 | 0.5929 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.7178 | 0.7004 | 0.7090 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.5996 | 0.2621 | 0.3648 | 54,828 |
| benign_degradation | 0.8662 | 0.6006 | 0.7094 | 270,642 |
| normal | 0.9867 | 0.9984 | 0.9925 | 10,570,307 |

#### `f20-xgboost-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.4628 | 0.9690 | 0.6264 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.2916 | 0.7254 | 0.4159 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.3019 | 0.7819 | 0.4356 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0515 | 0.8090 | 0.0969 | 54,828 |
| benign_degradation | 0.1286 | 0.7921 | 0.2213 | 270,642 |
| normal | 0.9973 | 0.7682 | 0.8679 | 10,570,307 |

#### `f20-xgboost-smote`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.7862 | 0.8812 | 0.8310 | 50,779 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.6723 | 0.6811 | 0.6767 | 63,963 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.7679 | 0.7081 | 0.7368 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.4900 | 0.3123 | 0.3815 | 54,828 |
| benign_degradation | 0.9293 | 0.5491 | 0.6903 | 270,642 |
| normal | 0.9861 | 0.9980 | 0.9920 | 10,570,307 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `f20-decision-tree-none` | `f20-logistic-regression-none` | yes | 11,057,478 | 10,615,415 | 168,207 | 12,466 | 261,390 | 180,673 |
| `f20-decision-tree-none` | `f20-random-forest-none-cap4m` | yes | 11,057,478 | 10,746,957 | 36,665 | 90,574 | 183,282 | 127,239 |
| `f20-decision-tree-none` | `f20-xgboost-downsample` | yes | 11,057,478 | 8,332,993 | 2,450,629 | 178,649 | 95,207 | 2,629,278 |
| `f20-decision-tree-none` | `f20-xgboost-smote` | yes | 11,057,478 | 10,755,259 | 28,363 | 81,304 | 192,552 | 109,667 |

