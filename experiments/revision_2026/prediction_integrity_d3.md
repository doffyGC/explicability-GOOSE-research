# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-13 06:51:27 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `d3-decision-tree-none` | full_grouped_run | decision-tree | none | — | 5 | 23,226,530 |
| `d3-decision-tree-none-cap4m` | full_grouped_run | decision-tree | none | 4,000,000 | 5 | 23,226,530 |
| `d3-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 23,226,530 |
| `d3-random-forest-none-cap4m` | full_grouped_run | random-forest | none | 4,000,000 | 5 | 23,226,530 |
| `d3-logistic-regression-none` | full_grouped_run | logistic-regression | none | — | 5 | 23,226,530 |
| `d3-decision-tree-downsample` | full_grouped_run | decision-tree | downsample | — | 5 | 23,226,530 |
| `d3-xgboost-downsample` | full_grouped_run | xgboost | downsample | — | 5 | 23,226,530 |
| `d3-random-forest-downsample` | full_grouped_run | random-forest | downsample | — | 5 | 23,226,530 |
| `d3-logistic-regression-downsample` | full_grouped_run | logistic-regression | downsample | — | 5 | 23,226,530 |
| `grouped-validation-full-smote` | full_grouped_run | decision-tree | smote | — | 5 | 23,226,530 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `d3-decision-tree-none`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.987973798672 | 0.987973798672 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.296479204554 | 0.296479204554 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.984260703325 | 0.984260703325 | pass |
| fold-01: recorded accuracy matches recomputation | 0.986811570911 | 0.986811570911 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.280787473512 | 0.280787473512 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982518428042 | 0.982518428042 | pass |
| fold-02: recorded accuracy matches recomputation | 0.977622678093 | 0.977622678093 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.262471137691 | 0.262471137691 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.970737813288 | 0.970737813288 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981388353564 | 0.981388353564 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.279171270132 | 0.279171270132 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.975861483106 | 0.975861483106 | pass |
| fold-04: recorded accuracy matches recomputation | 0.984508689005 | 0.984508689005 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.207738388374 | 0.207738388374 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.977197548918 | 0.977197548918 | pass |

</details>

### `d3-decision-tree-none-cap4m`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.987934417919 | 0.987934417919 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.296309259320 | 0.296309259320 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.984223641284 | 0.984223641284 | pass |
| fold-01: recorded accuracy matches recomputation | 0.986736015281 | 0.986736015281 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.279543636229 | 0.279543636229 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982404050289 | 0.982404050289 | pass |
| fold-02: recorded accuracy matches recomputation | 0.977627502688 | 0.977627502688 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.262701410137 | 0.262701410137 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.970771063198 | 0.970771063198 | pass |
| fold-03: recorded accuracy matches recomputation | 0.981385924258 | 0.981385924258 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.279336437068 | 0.279336437068 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.975874748925 | 0.975874748925 | pass |
| fold-04: recorded accuracy matches recomputation | 0.984470715207 | 0.984470715207 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.209308649278 | 0.209308649278 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.977205689175 | 0.977205689175 | pass |

</details>

### `d3-xgboost-none`

35 checks, 0 failed.

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

### `d3-random-forest-none-cap4m`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.986814578398 | 0.986814578398 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.335879329845 | 0.335879329845 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.984159483024 | 0.984159483024 | pass |
| fold-01: recorded accuracy matches recomputation | 0.985432911589 | 0.985432911589 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.316668673170 | 0.316668673170 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982365069997 | 0.982365069997 | pass |
| fold-02: recorded accuracy matches recomputation | 0.975659351824 | 0.975659351824 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.304252713604 | 0.304252713604 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.970632257234 | 0.970632257234 | pass |
| fold-03: recorded accuracy matches recomputation | 0.979927368592 | 0.979927368592 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.316816784247 | 0.316816784247 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.975970588422 | 0.975970588422 | pass |
| fold-04: recorded accuracy matches recomputation | 0.981942709530 | 0.981942709530 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.255704186038 | 0.255704186038 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.976855589231 | 0.976855589231 | pass |

</details>

### `d3-logistic-regression-none`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.980586909352 | 0.980586909352 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.165033092873 | 0.165033092873 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.970994197197 | 0.970994197197 | pass |
| fold-01: recorded accuracy matches recomputation | 0.982057108233 | 0.982057108233 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.165157889776 | 0.165157889776 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.973180973452 | 0.973180973452 | pass |
| fold-02: recorded accuracy matches recomputation | 0.971138706532 | 0.971138706532 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.164226343435 | 0.164226343435 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.956938368266 | 0.956938368266 | pass |
| fold-03: recorded accuracy matches recomputation | 0.974363041890 | 0.974363041890 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.164507674947 | 0.164507674947 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.961729954775 | 0.961729954775 | pass |
| fold-04: recorded accuracy matches recomputation | 0.984139693611 | 0.984139693611 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.165334409464 | 0.165334409464 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.976290030752 | 0.976290030752 | pass |

</details>

### `d3-decision-tree-downsample`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.563167295111 | 0.563167295111 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.217521034573 | 0.217521034573 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.709850636260 | 0.709850636260 | pass |
| fold-01: recorded accuracy matches recomputation | 0.537908790131 | 0.537908790131 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.210428885931 | 0.210428885931 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.689630883907 | 0.689630883907 | pass |
| fold-02: recorded accuracy matches recomputation | 0.478001550682 | 0.478001550682 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.193061122559 | 0.193061122559 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.633129457273 | 0.633129457273 | pass |
| fold-03: recorded accuracy matches recomputation | 0.502497205690 | 0.502497205690 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.227967213084 | 0.227967213084 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.655898370290 | 0.655898370290 | pass |
| fold-04: recorded accuracy matches recomputation | 0.506475282055 | 0.506475282055 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.153984094741 | 0.153984094741 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.660879335093 | 0.660879335093 | pass |

</details>

### `d3-xgboost-downsample`

35 checks, 0 failed.

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

### `d3-random-forest-downsample`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.618427567046 | 0.618427567046 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.201870859218 | 0.201870859218 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.751157844794 | 0.751157844794 | pass |
| fold-01: recorded accuracy matches recomputation | 0.591676873142 | 0.591676873142 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.194125796914 | 0.194125796914 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.731363605361 | 0.731363605361 | pass |
| fold-02: recorded accuracy matches recomputation | 0.517835391638 | 0.517835391638 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.175172731207 | 0.175172731207 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.664916016173 | 0.664916016173 | pass |
| fold-03: recorded accuracy matches recomputation | 0.537517360432 | 0.537517360432 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.194189080987 | 0.194189080987 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.681753268576 | 0.681753268576 | pass |
| fold-04: recorded accuracy matches recomputation | 0.519192507170 | 0.519192507170 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.150555708166 | 0.150555708166 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.672388647357 | 0.672388647357 | pass |

</details>

### `d3-logistic-regression-downsample`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.474830431825 | 0.474830431825 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.146301245333 | 0.146301245333 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.631792900447 | 0.631792900447 | pass |
| fold-01: recorded accuracy matches recomputation | 0.482754473762 | 0.482754473762 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.140021744874 | 0.140021744874 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.639578114706 | 0.639578114706 | pass |
| fold-02: recorded accuracy matches recomputation | 0.391644142431 | 0.391644142431 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.126655657692 | 0.126655657692 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.546297540997 | 0.546297540997 | pass |
| fold-03: recorded accuracy matches recomputation | 0.421253420160 | 0.421253420160 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.136506431806 | 0.136506431806 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.576330866559 | 0.576330866559 | pass |
| fold-04: recorded accuracy matches recomputation | 0.410959687815 | 0.410959687815 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.118557845198 | 0.118557845198 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.572418648620 | 0.572418648620 | pass |

</details>

### `grouped-validation-full-smote`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.984384153898 | 0.984384153898 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.273863510723 | 0.273863510723 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.980921755253 | 0.980921755253 | pass |
| fold-01: recorded accuracy matches recomputation | 0.986431575972 | 0.986431575972 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.278756854939 | 0.278756854939 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.982234346238 | 0.982234346238 | pass |
| fold-02: recorded accuracy matches recomputation | 0.977248913899 | 0.977248913899 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.261548600807 | 0.261548600807 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.970517398559 | 0.970517398559 | pass |
| fold-03: recorded accuracy matches recomputation | 0.980590569286 | 0.980590569286 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.276082693535 | 0.276082693535 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.975156036007 | 0.975156036007 | pass |
| fold-04: recorded accuracy matches recomputation | 0.984290839321 | 0.984290839321 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.203269478254 | 0.203269478254 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.977019037287 | 0.977019037287 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d3-decision-tree-none` | 0.9844 | 0.3702 | 0.2635 | 0.2791 | 0.9756 | 0.9844 | 0.9792 |
| `d3-decision-tree-none-cap4m` | 0.9843 | 0.3258 | 0.2632 | 0.2789 | 0.9749 | 0.9843 | 0.9792 |
| `d3-xgboost-none` | 0.9847 | 0.3056 | 0.2665 | 0.2823 | 0.9751 | 0.9847 | 0.9796 |
| `d3-random-forest-none-cap4m` | 0.9827 | 0.3964 | 0.2883 | 0.3176 | 0.9764 | 0.9827 | 0.9790 |
| `d3-logistic-regression-none` | 0.9790 | 0.1636 | 0.1667 | 0.1649 | 0.9585 | 0.9790 | 0.9686 |
| `d3-decision-tree-downsample` | 0.5238 | 0.2321 | 0.6466 | 0.2076 | 0.9808 | 0.5238 | 0.6756 |
| `d3-xgboost-downsample` | 0.5641 | 0.2253 | 0.6721 | 0.2099 | 0.9806 | 0.5641 | 0.7085 |
| `d3-random-forest-downsample` | 0.5655 | 0.2053 | 0.6402 | 0.1874 | 0.9789 | 0.5655 | 0.7081 |
| `d3-logistic-regression-downsample` | 0.4436 | 0.1840 | 0.5161 | 0.1351 | 0.9775 | 0.4436 | 0.6012 |
| `grouped-validation-full-smote` | 0.9831 | 0.3015 | 0.2521 | 0.2706 | 0.9739 | 0.9831 | 0.9780 |

### Per-class (pooled over folds)

#### `d3-decision-tree-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.1250 | 0.0000 | 0.0000 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.2857 | 0.0000 | 0.0001 | 54,828 |
| benign_degradation | 0.8249 | 0.5823 | 0.6827 | 270,680 |
| normal | 0.9857 | 0.9985 | 0.9921 | 22,739,305 |

#### `d3-decision-tree-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0667 | 0.0000 | 0.0001 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0556 | 0.0000 | 0.0001 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0233 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8234 | 0.5808 | 0.6811 | 270,680 |
| normal | 0.9857 | 0.9985 | 0.9921 | 22,739,305 |

#### `d3-xgboost-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0039 | 0.0000 | 0.0000 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8438 | 0.6003 | 0.7015 | 270,680 |
| normal | 0.9859 | 0.9987 | 0.9922 | 22,739,305 |

#### `d3-random-forest-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0362 | 0.0116 | 0.0176 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.1543 | 0.0406 | 0.0643 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.1652 | 0.0357 | 0.0587 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.2202 | 0.0565 | 0.0899 | 54,828 |
| benign_degradation | 0.8161 | 0.5888 | 0.6841 | 270,680 |
| normal | 0.9863 | 0.9964 | 0.9913 | 22,739,305 |

#### `d3-logistic-regression-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0027 | 0.0000 | 0.0000 | 270,680 |
| normal | 0.9790 | 1.0000 | 0.9894 | 22,739,305 |

#### `d3-decision-tree-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0348 | 0.9156 | 0.0670 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0119 | 0.5718 | 0.0233 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0090 | 0.6365 | 0.0177 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0107 | 0.6026 | 0.0210 | 54,828 |
| benign_degradation | 0.3283 | 0.6319 | 0.4321 | 270,680 |
| normal | 0.9978 | 0.5211 | 0.6847 | 22,739,305 |

#### `d3-xgboost-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0370 | 0.9024 | 0.0712 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0126 | 0.6292 | 0.0248 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0124 | 0.6765 | 0.0243 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0112 | 0.5716 | 0.0221 | 54,828 |
| benign_degradation | 0.2801 | 0.6916 | 0.3988 | 270,680 |
| normal | 0.9981 | 0.5613 | 0.7186 | 22,739,305 |

#### `d3-random-forest-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0367 | 0.6606 | 0.0696 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0121 | 0.6217 | 0.0238 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0118 | 0.6626 | 0.0233 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0144 | 0.6217 | 0.0282 | 54,828 |
| benign_degradation | 0.1587 | 0.7115 | 0.2595 | 270,680 |
| normal | 0.9978 | 0.5630 | 0.7199 | 22,739,305 |

#### `d3-logistic-regression-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0195 | 0.7368 | 0.0381 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0079 | 0.3322 | 0.0155 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0064 | 0.5378 | 0.0127 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0127 | 0.5383 | 0.0248 | 54,828 |
| benign_degradation | 0.0598 | 0.5093 | 0.1070 | 270,680 |
| normal | 0.9976 | 0.4420 | 0.6126 | 22,739,305 |

#### `grouped-validation-full-smote`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0007 | 0.0001 | 0.0001 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0014 | 0.0000 | 0.0000 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0053 | 0.0012 | 0.0019 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8169 | 0.5131 | 0.6303 | 270,680 |
| normal | 0.9850 | 0.9980 | 0.9915 | 22,739,305 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `d3-decision-tree-none` | `d3-decision-tree-none-cap4m` | yes | 23,226,530 | 22,860,241 | 3,219 | 2,422 | 360,648 | 5,641 |
| `d3-decision-tree-none` | `d3-xgboost-none` | yes | 23,226,530 | 22,854,835 | 8,625 | 16,612 | 346,458 | 25,237 |
| `d3-decision-tree-none` | `d3-random-forest-none-cap4m` | yes | 23,226,530 | 22,799,289 | 64,171 | 26,352 | 336,718 | 90,523 |
| `d3-decision-tree-none` | `d3-logistic-regression-none` | yes | 23,226,530 | 22,705,439 | 158,021 | 33,460 | 329,610 | 191,481 |
| `d3-decision-tree-none` | `d3-decision-tree-downsample` | yes | 23,226,530 | 11,995,484 | 10,867,976 | 171,450 | 191,620 | 11,039,426 |
| `d3-decision-tree-none` | `d3-xgboost-downsample` | yes | 23,226,530 | 12,920,303 | 9,943,157 | 180,763 | 182,307 | 10,123,920 |
| `d3-decision-tree-none` | `d3-random-forest-downsample` | yes | 23,226,530 | 12,960,420 | 9,903,040 | 173,978 | 189,092 | 10,077,018 |
| `d3-decision-tree-none` | `d3-logistic-regression-downsample` | yes | 23,226,530 | 10,166,405 | 12,697,055 | 135,833 | 227,237 | 12,832,888 |
| `d3-decision-tree-none` | `grouped-validation-full-smote` | yes | 23,226,530 | 22,828,808 | 34,652 | 4,972 | 358,098 | 39,624 |

