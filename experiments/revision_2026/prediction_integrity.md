# Prediction integrity audit - Gray-GOOSE (checklist E.4/E.5)

- Generated: 2026-09-13 07:15:11 UTC
- Recomputed independently of `run_grouped_validation.py`: every number below
  comes from a `numpy.bincount` confusion matrix over the persisted
  `grouped_predictions.csv`, not from the `sklearn` helpers the runner used.

## 1. Runs audited

| run | status | model | balance | train cap | folds | rows |
|---|---|---|---|---:|---:|---:|
| `d3-decision-tree-none` | full_grouped_run | decision-tree | none | — | 5 | 23,226,530 |
| `d3-decision-tree-downsample` | full_grouped_run | decision-tree | downsample | — | 5 | 23,226,530 |
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
| `d3-decision-tree-downsample` | 0.5238 | 0.2321 | 0.6466 | 0.2076 | 0.9808 | 0.5238 | 0.6756 |
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

#### `d3-decision-tree-downsample`

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| DETERMINISTIC_BURST_ORIENTEDGRAYHOLE | 0.0348 | 0.9156 | 0.0670 | 50,782 |
| FULLY_RANDOMIZED_ORIENTEDGRAYHOLE | 0.0119 | 0.5718 | 0.0233 | 63,976 |
| RANDOMIC_BURST_ORIENTEDGRAYHOLE | 0.0090 | 0.6365 | 0.0177 | 46,959 |
| RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE | 0.0107 | 0.6026 | 0.0210 | 54,828 |
| benign_degradation | 0.3283 | 0.6319 | 0.4321 | 270,680 |
| normal | 0.9978 | 0.5211 | 0.6847 | 22,739,305 |

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
| `d3-decision-tree-none` | `d3-decision-tree-downsample` | yes | 23,226,530 | 11,995,484 | 10,867,976 | 171,450 | 191,620 | 11,039,426 |
| `d3-decision-tree-none` | `grouped-validation-full-smote` | yes | 23,226,530 | 22,828,808 | 34,652 | 4,972 | 358,098 | 39,624 |

