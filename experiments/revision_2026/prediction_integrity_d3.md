# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-12 20:58:37 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `d3-decision-tree-none` | full_grouped_run | decision-tree | none | — | 5 | 20,796,921 |
| `d3-decision-tree-none-cap4m` | full_grouped_run | decision-tree | none | 4,000,000 | 5 | 20,796,921 |
| `d3-xgboost-none` | full_grouped_run | xgboost | none | — | 5 | 20,796,921 |
| `d3-random-forest-none-cap4m` | full_grouped_run | random-forest | none | 4,000,000 | 5 | 20,796,921 |
| `d3-logistic-regression-none` | full_grouped_run | logistic-regression | none | — | 5 | 20,796,921 |
| `d3-decision-tree-downsample` | full_grouped_run | decision-tree | downsample | — | 5 | 20,796,921 |
| `d3-xgboost-downsample` | full_grouped_run | xgboost | downsample | — | 5 | 20,796,921 |
| `d3-random-forest-downsample` | full_grouped_run | random-forest | downsample | — | 5 | 20,796,921 |
| `d3-logistic-regression-downsample` | full_grouped_run | logistic-regression | downsample | — | 5 | 20,796,921 |

## 2. Count reconciliation

Checklist E.5: class sums and paired prediction counts, checked before any
statistical test is written.

### `d3-decision-tree-none`

35 checks, 0 failed.

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
| fold-00: recorded weighted_f1 matches recomputation | 0.983555825684 | 0.983555825684 | pass |
| fold-01: recorded accuracy matches recomputation | 0.991661242079 | 0.991661242079 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.312134197122 | 0.312134197122 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.988749489295 | 0.988749489295 | pass |
| fold-02: recorded accuracy matches recomputation | 0.985178253004 | 0.985178253004 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.277688959121 | 0.277688959121 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.981437602706 | 0.981437602706 | pass |
| fold-03: recorded accuracy matches recomputation | 0.973625854646 | 0.973625854646 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.261737131411 | 0.261737131411 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.967021131065 | 0.967021131065 | pass |
| fold-04: recorded accuracy matches recomputation | 0.986145105488 | 0.986145105488 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.260803886658 | 0.260803886658 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.981618676736 | 0.981618676736 | pass |

</details>

### `d3-decision-tree-none-cap4m`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.987323392684 | 0.987323392684 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.266551507895 | 0.266551507895 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.983543839624 | 0.983543839624 | pass |
| fold-01: recorded accuracy matches recomputation | 0.991653718711 | 0.991653718711 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.312088043060 | 0.312088043060 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.988742681330 | 0.988742681330 | pass |
| fold-02: recorded accuracy matches recomputation | 0.985191234453 | 0.985191234453 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.278396086121 | 0.278396086121 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.981506021567 | 0.981506021567 | pass |
| fold-03: recorded accuracy matches recomputation | 0.973563923810 | 0.973563923810 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.261314068234 | 0.261314068234 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.966931465156 | 0.966931465156 | pass |
| fold-04: recorded accuracy matches recomputation | 0.986130205463 | 0.986130205463 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.261233121044 | 0.261233121044 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.981638026315 | 0.981638026315 | pass |

</details>

### `d3-xgboost-none`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.987691475884 | 0.987691475884 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.269189626313 | 0.269189626313 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.983875046291 | 0.983875046291 | pass |
| fold-01: recorded accuracy matches recomputation | 0.991966654999 | 0.991966654999 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.314738276886 | 0.314738276886 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.989079349768 | 0.989079349768 | pass |
| fold-02: recorded accuracy matches recomputation | 0.984731455114 | 0.984731455114 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.263316364084 | 0.263316364084 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.979951136932 | 0.979951136932 | pass |
| fold-03: recorded accuracy matches recomputation | 0.974701852971 | 0.974701852971 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.267042262789 | 0.267042262789 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.968292795187 | 0.968292795187 | pass |
| fold-04: recorded accuracy matches recomputation | 0.986064751783 | 0.986064751783 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.259437054538 | 0.259437054538 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.981493215114 | 0.981493215114 | pass |

</details>

### `d3-random-forest-none-cap4m`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.986391311031 | 0.986391311031 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.289269645185 | 0.289269645185 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.983424644811 | 0.983424644811 | pass |
| fold-01: recorded accuracy matches recomputation | 0.990395883213 | 0.990395883213 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.332257902979 | 0.332257902979 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.988168455288 | 0.988168455288 | pass |
| fold-02: recorded accuracy matches recomputation | 0.984606833199 | 0.984606833199 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.306593817980 | 0.306593817980 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.981719884063 | 0.981719884063 | pass |
| fold-03: recorded accuracy matches recomputation | 0.972900664538 | 0.972900664538 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.294112384552 | 0.294112384552 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.967611606483 | 0.967611606483 | pass |
| fold-04: recorded accuracy matches recomputation | 0.985046494729 | 0.985046494729 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.294759520493 | 0.294759520493 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.981860293658 | 0.981860293658 | pass |

</details>

### `d3-logistic-regression-none`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.984134681134 | 0.984134681134 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.165341481806 | 0.165341481806 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.976283751902 | 0.976283751902 | pass |
| fold-01: recorded accuracy matches recomputation | 0.982785279850 | 0.982785279850 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.165219651675 | 0.165219651675 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.974265612449 | 0.974265612449 | pass |
| fold-02: recorded accuracy matches recomputation | 0.978761404498 | 0.978761404498 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.164877787063 | 0.164877787063 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.968274065581 | 0.968274065581 | pass |
| fold-03: recorded accuracy matches recomputation | 0.966227716886 | 0.966227716886 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.163803969837 | 0.163803969837 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.949648500464 | 0.949648500464 | pass |
| fold-04: recorded accuracy matches recomputation | 0.982464799357 | 0.982464799357 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.165192474821 | 0.165192474821 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.973792418919 | 0.973792418919 | pass |

</details>

### `d3-decision-tree-downsample`

35 checks, 0 failed.

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
| fold-00: recorded weighted_f1 matches recomputation | 0.697729742137 | 0.697729742137 | pass |
| fold-01: recorded accuracy matches recomputation | 0.558658089072 | 0.558658089072 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.204966839720 | 0.204966839720 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.705941071361 | 0.705941071361 | pass |
| fold-02: recorded accuracy matches recomputation | 0.524415864275 | 0.524415864275 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.204858981772 | 0.204858981772 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.677787975417 | 0.677787975417 | pass |
| fold-03: recorded accuracy matches recomputation | 0.441600021096 | 0.441600021096 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.200140965270 | 0.200140965270 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.596943721587 | 0.596943721587 | pass |
| fold-04: recorded accuracy matches recomputation | 0.496356544823 | 0.496356544823 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.172889275658 | 0.172889275658 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.653838242237 | 0.653838242237 | pass |

</details>

### `d3-xgboost-downsample`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.586860744330 | 0.586860744330 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.193009622382 | 0.193009622382 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.730116732749 | 0.730116732749 | pass |
| fold-01: recorded accuracy matches recomputation | 0.617519667338 | 0.617519667338 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.218598423555 | 0.218598423555 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.753212775278 | 0.753212775278 | pass |
| fold-02: recorded accuracy matches recomputation | 0.560367870676 | 0.560367870676 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.208985308452 | 0.208985308452 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.707881406774 | 0.707881406774 | pass |
| fold-03: recorded accuracy matches recomputation | 0.440779737190 | 0.440779737190 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.185467157662 | 0.185467157662 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.593787056625 | 0.593787056625 | pass |
| fold-04: recorded accuracy matches recomputation | 0.515612166190 | 0.515612166190 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.179870345324 | 0.179870345324 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.670747531831 | 0.670747531831 | pass |

</details>

### `d3-random-forest-downsample`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.581721149412 | 0.581721149412 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.173017710845 | 0.173017710845 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.725328758356 | 0.725328758356 | pass |
| fold-01: recorded accuracy matches recomputation | 0.621619902963 | 0.621619902963 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.197108286828 | 0.197108286828 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.755017121037 | 0.755017121037 | pass |
| fold-02: recorded accuracy matches recomputation | 0.564517450018 | 0.564517450018 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.186181430771 | 0.186181430771 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.709125047978 | 0.709125047978 | pass |
| fold-03: recorded accuracy matches recomputation | 0.443979363846 | 0.443979363846 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.165681524653 | 0.165681524653 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.595537293639 | 0.595537293639 | pass |
| fold-04: recorded accuracy matches recomputation | 0.525519086267 | 0.525519086267 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.161073749465 | 0.161073749465 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.678158333256 | 0.678158333256 | pass |

</details>

### `d3-logistic-regression-downsample`

35 checks, 0 failed.

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
| fold-00: recorded accuracy matches recomputation | 0.484908694799 | 0.484908694799 | pass |
| fold-00: recorded macro_f1 matches recomputation | 0.136177609199 | 0.136177609199 | pass |
| fold-00: recorded weighted_f1 matches recomputation | 0.643335949616 | 0.643335949616 | pass |
| fold-01: recorded accuracy matches recomputation | 0.543724024228 | 0.543724024228 | pass |
| fold-01: recorded macro_f1 matches recomputation | 0.156698795738 | 0.156698795738 | pass |
| fold-01: recorded weighted_f1 matches recomputation | 0.693470296339 | 0.693470296339 | pass |
| fold-02: recorded accuracy matches recomputation | 0.470122485877 | 0.470122485877 | pass |
| fold-02: recorded macro_f1 matches recomputation | 0.139238982188 | 0.139238982188 | pass |
| fold-02: recorded weighted_f1 matches recomputation | 0.627349402900 | 0.627349402900 | pass |
| fold-03: recorded accuracy matches recomputation | 0.351356005613 | 0.351356005613 | pass |
| fold-03: recorded macro_f1 matches recomputation | 0.123348826497 | 0.123348826497 | pass |
| fold-03: recorded weighted_f1 matches recomputation | 0.499773525626 | 0.499773525626 | pass |
| fold-04: recorded accuracy matches recomputation | 0.419454323841 | 0.419454323841 | pass |
| fold-04: recorded macro_f1 matches recomputation | 0.122169100572 | 0.122169100572 | pass |
| fold-04: recorded weighted_f1 matches recomputation | 0.580504087847 | 0.580504087847 | pass |

</details>

## 3. Metrics, explicitly labelled

Checklist E.4. `accuracy` is the overall (micro) figure; `macro` averages
the per-class values without weights, so each of the four rare attack
classes counts as much as `normal`; `weighted` averages the same per-class
values by support, so it tracks the majority class. Pooled over all folds,
on the original (never rebalanced) test distribution.

| run | accuracy (micro) | macro P | macro R | macro F1 | weighted P | weighted R | weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `d3-decision-tree-none` | 0.9862 | 0.3014 | 0.2640 | 0.2794 | 0.9789 | 0.9862 | 0.9823 |
| `d3-decision-tree-none-cap4m` | 0.9862 | 0.3009 | 0.2644 | 0.2795 | 0.9789 | 0.9862 | 0.9823 |
| `d3-xgboost-none` | 0.9864 | 0.3085 | 0.2598 | 0.2788 | 0.9792 | 0.9864 | 0.9823 |
| `d3-random-forest-none-cap4m` | 0.9852 | 0.3677 | 0.2807 | 0.3055 | 0.9802 | 0.9852 | 0.9822 |
| `d3-logistic-regression-none` | 0.9802 | 0.1639 | 0.1667 | 0.1650 | 0.9609 | 0.9802 | 0.9704 |
| `d3-decision-tree-downsample` | 0.5234 | 0.2263 | 0.6530 | 0.1993 | 0.9826 | 0.5234 | 0.6766 |
| `d3-xgboost-downsample` | 0.5592 | 0.2213 | 0.6608 | 0.2008 | 0.9825 | 0.5592 | 0.7063 |
| `d3-random-forest-downsample` | 0.5622 | 0.2026 | 0.5789 | 0.1799 | 0.9806 | 0.5622 | 0.7074 |
| `d3-logistic-regression-downsample` | 0.4698 | 0.1841 | 0.5096 | 0.1386 | 0.9793 | 0.4698 | 0.6272 |

### Per-class (pooled over folds)

#### `d3-decision-tree-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8209 | 0.5855 | 0.6835 | 270,680 |
| normal | 0.9877 | 0.9983 | 0.9930 | 20,385,924 |

#### `d3-decision-tree-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8173 | 0.5882 | 0.6841 | 270,680 |
| normal | 0.9878 | 0.9983 | 0.9930 | 20,385,924 |

#### `d3-xgboost-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.8638 | 0.5602 | 0.6796 | 270,680 |
| normal | 0.9874 | 0.9988 | 0.9931 | 20,385,924 |

#### `d3-random-forest-none-cap4m`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0032 | 0.0009 | 0.0014 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0122 | 0.0014 | 0.0025 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.1689 | 0.0379 | 0.0619 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.2209 | 0.0570 | 0.0906 | 54,828 |
| benign_degradation | 0.8130 | 0.5901 | 0.6838 | 270,680 |
| normal | 0.9882 | 0.9970 | 0.9926 | 20,385,924 |

#### `d3-logistic-regression-none`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0000 | 0.0000 | 0.0000 | 54,828 |
| benign_degradation | 0.0030 | 0.0000 | 0.0000 | 270,680 |
| normal | 0.9802 | 1.0000 | 0.9900 | 20,385,924 |

#### `d3-decision-tree-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0150 | 0.9208 | 0.0294 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0050 | 0.5641 | 0.0099 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0088 | 0.6392 | 0.0174 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0123 | 0.6088 | 0.0240 | 54,828 |
| benign_degradation | 0.3183 | 0.6644 | 0.4304 | 270,680 |
| normal | 0.9982 | 0.5207 | 0.6844 | 20,385,924 |

#### `d3-xgboost-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0151 | 0.8975 | 0.0296 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0046 | 0.5971 | 0.0091 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0127 | 0.6651 | 0.0249 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0123 | 0.5640 | 0.0241 | 54,828 |
| benign_degradation | 0.2847 | 0.6843 | 0.4021 | 270,680 |
| normal | 0.9985 | 0.5570 | 0.7151 | 20,385,924 |

#### `d3-random-forest-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0105 | 0.4964 | 0.0206 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0044 | 0.5952 | 0.0088 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0113 | 0.6109 | 0.0221 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0136 | 0.5108 | 0.0266 | 54,828 |
| benign_degradation | 0.1776 | 0.6996 | 0.2833 | 270,680 |
| normal | 0.9980 | 0.5604 | 0.7178 | 20,385,924 |

#### `d3-logistic-regression-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0076 | 0.6879 | 0.0149 | 17,094 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0033 | 0.3380 | 0.0066 | 21,436 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0068 | 0.5212 | 0.0134 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0143 | 0.5307 | 0.0279 | 54,828 |
| benign_degradation | 0.0748 | 0.5111 | 0.1305 | 270,680 |
| normal | 0.9980 | 0.4689 | 0.6380 | 20,385,924 |

## 4. Paired predictions across runs

Two runs are pairable only if they predicted exactly the same rows with the
same ground truth. `discordant` is the number of rows where exactly one of
the two got it right - the only cells a McNemar-style paired test consumes.

| run A | run B | pairable | n paired | both correct | only A | only B | neither | discordant |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `d3-decision-tree-none` | `d3-decision-tree-none-cap4m` | yes | 20,796,921 | 20,506,942 | 2,866 | 2,564 | 284,549 | 5,430 |
| `d3-decision-tree-none` | `d3-xgboost-none` | yes | 20,796,921 | 20,491,195 | 18,613 | 22,448 | 264,665 | 41,061 |
| `d3-decision-tree-none` | `d3-random-forest-none-cap4m` | yes | 20,796,921 | 20,465,386 | 44,422 | 24,493 | 262,620 | 68,915 |
| `d3-decision-tree-none` | `d3-logistic-regression-none` | yes | 20,796,921 | 20,350,980 | 158,828 | 34,599 | 252,514 | 193,427 |
| `d3-decision-tree-none` | `d3-decision-tree-downsample` | yes | 20,796,921 | 10,768,091 | 9,741,717 | 117,232 | 169,881 | 9,858,949 |
| `d3-decision-tree-none` | `d3-xgboost-downsample` | yes | 20,796,921 | 11,511,024 | 8,998,784 | 119,394 | 167,719 | 9,118,178 |
| `d3-decision-tree-none` | `d3-random-forest-downsample` | yes | 20,796,921 | 11,581,785 | 8,928,023 | 110,106 | 177,007 | 9,038,129 |
| `d3-decision-tree-none` | `d3-logistic-regression-downsample` | yes | 20,796,921 | 9,675,236 | 10,834,572 | 94,811 | 192,302 | 10,929,383 |

