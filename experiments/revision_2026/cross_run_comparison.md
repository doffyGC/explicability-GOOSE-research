# Cross-run comparison (checklist D.5)

- Generated: 2026-09-21 16:44:44 UTC
- Runs: **24**, all on dataset `3109e4d48052…`, protocol `stratified-group-kfold`, 5 folds from `splits_grouped.json`.
- Rows evaluated per run: **11,057,478**

Every metric here is **pooled over the folds** - counts summed first, metric computed once - not averaged across them. The folds partition the runs rather than replicate them, so a fold mean would weight a fold holding one `SAG.DB` run the same as one holding four (`bootstrap_run_intervals.py`). These are the same point estimates that script reports.

**This table ranks nothing.** It carries no interval, and two rows differing by less than their unshown uncertainty are not two results. Every comparison this repository makes is paired over the same runs - `grouped_pr_curves.py` for the ranking axis, `bootstrap_run_intervals.py` for macro F1 - and those are the only places a difference between two rows may be read from.

## Headline

AP is on `ANY_ATTACK` and is the axis card D compares on (§11); macro F1 and accuracy sit at whatever threshold each run's training prior implies and are context, not verdict.

| Card | Run | Model | Scenario | Variant | AP `ANY_ATTACK` [95% CI] | macro F1 | weighted F1 | accuracy |
|---|---|---|---|---|---|---:|---:|---:|
| D.1 | `d1-xgboost-no-absolute-time` | `xgboost` | none | features: `no-absolute-time` | 0.8304 [0.8011, 0.8597] | 0.7279 | 0.9791 | 0.9814 |
| D.1 | `d1-xgboost-no-counters` | `xgboost` | none | features: `no-counters` | 0.8306 [0.8012, 0.8596] | 0.7263 | 0.9787 | 0.9810 |
| D.1 | `d1-xgboost-no-delta` | `xgboost` | none | features: `no-delta` | 0.1101 [0.0906, 0.1344] | 0.2275 | 0.9460 | 0.9613 |
| D.1 | `d1-xgboost-no-electrical` | `xgboost` | none | features: `no-electrical` | 0.8327 [0.8037, 0.8616] | 0.7288 | 0.9788 | 0.9811 |
| D.1 | `d1-xgboost-no-goose-header` | `xgboost` | none | features: `no-goose-header` | 0.8327 [0.8036, 0.8619] | 0.7304 | 0.9792 | 0.9815 |
| D.1 | `d1-xgboost-no-sequence` | `xgboost` | none | features: `no-sequence` | 0.8240 [0.7934, 0.8531] | 0.7155 | 0.9779 | 0.9805 |
| D.1 | `d1-xgboost-no-size-state-deltas` | `xgboost` | none | features: `no-size-state-deltas` | 0.8325 [0.8032, 0.8616] | 0.7296 | 0.9792 | 0.9814 |
| D.1 | `d1-xgboost-no-timing-deltas` | `xgboost` | none | features: `no-timing-deltas` | 0.7683 [0.7296, 0.8021] | 0.6450 | 0.9702 | 0.9745 |
| D.2 | `d2-rule-delay` | `rule:delay` | none | rule: `delay` | 0.0175 [0.0147, 0.0211] | 0.0769 | 0.4266 | 0.2809 |
| D.2 | `d2-rule-interval-t` | `rule:interval-t` | none | rule: `interval-t` | 0.1010 [0.0822, 0.1237] | 0.1621 | 0.9138 | 0.9053 |
| D.2 | `d2-rule-interval-timestamp` | `rule:interval-timestamp` | none | rule: `interval-timestamp` | 0.2411 [0.1895, 0.3052] | 0.2350 | 0.9431 | 0.9597 |
| D.2 | `d2-rule-sqnum-gap` | `rule:sqnum-gap` | none | rule: `sqnum-gap` | 0.0451 [0.0338, 0.0592] | 0.1866 | 0.9215 | 0.9214 |
| D.2 | `d2-rule-stnum-gap` | `rule:stnum-gap` | none | rule: `stnum-gap` | 0.2192 [0.1754, 0.2712] | 0.1647 | 0.9334 | 0.9451 |
| D.2 | `d2-rule-time-since-change` | `rule:time-since-change` | none | rule: `time-since-change` | 0.0127 [0.0105, 0.0153] | 0.0019 | 0.0001 | 0.0058 |
| D.3 | `v2-decision-tree-downsample` | `decision-tree` | downsample | — | 0.6664 [0.6231, 0.7086] | 0.4376 | 0.8374 | 0.7608 |
| D.3 | `v2-decision-tree-none` | `decision-tree` | none | — | 0.7572 [0.7209, 0.7944] | 0.7043 | 0.9771 | 0.9799 |
| D.3 | `v2-decision-tree-none-cap4m` | `decision-tree` | none, cap 4,000,000 | — | 0.7548 [0.7178, 0.7917] | 0.7033 | 0.9770 | 0.9798 |
| D.3 | `v2-logistic-regression-downsample` | `logistic-regression` | downsample | — | 0.4593 [0.3961, 0.5186] | 0.3131 | 0.7939 | 0.6931 |
| D.3 | `v2-logistic-regression-none` | `logistic-regression` | none | — | 0.5087 [0.4484, 0.5630] | 0.3677 | 0.9496 | 0.9622 |
| D.3 | `v2-random-forest-downsample` | `random-forest` | downsample | — | 0.3758 [0.3434, 0.4101] | 0.4489 | 0.8653 | 0.8048 |
| D.3 | `v2-random-forest-none-cap4m` | `random-forest` | none, cap 4,000,000 | — | 0.7934 [0.7612, 0.8237] | 0.7355 | 0.9799 | 0.9815 |
| D.3 | `v2-xgboost-downsample` | `xgboost` | downsample | — | 0.8208 [0.7885, 0.8518] | 0.4594 | 0.8548 | 0.7880 |
| D.3 | `v2-xgboost-none` | `xgboost` | none | — | 0.8329 [0.8037, 0.8618] | 0.7310 | 0.9793 | 0.9815 |
| D.4 | `d4-xgboost-tuned` | `xgboost` | none | grid: `champion-xgboost` | 0.8328 [0.8037, 0.8619] | 0.7295 | 0.9791 | 0.9814 |

Cards: **D.1** feature-group ablation · **D.2** rule baseline · **D.3** model family comparison · **D.4** nested tuning. Each run's card is derived from its own report - the ablated feature set, the `rule` block, the `tuned` block - never from its directory name.

## Per class

Per-class values first, macro as the headline average, weighted and accuracy as context (`ablations_baselines.md` §5).

### `SAG.DB`

50,779 rows across all folds.

| Run | precision | recall | F1 |
|---|---:|---:|---:|
| `d1-xgboost-no-absolute-time` | 0.8259 | 0.8871 | 0.8554 |
| `d1-xgboost-no-counters` | 0.8228 | 0.8826 | 0.8516 |
| `d1-xgboost-no-delta` | 0.1834 | 0.0090 | 0.0172 |
| `d1-xgboost-no-electrical` | 0.8295 | 0.8823 | 0.8551 |
| `d1-xgboost-no-goose-header` | 0.8290 | 0.8826 | 0.8550 |
| `d1-xgboost-no-sequence` | 0.8147 | 0.8685 | 0.8408 |
| `d1-xgboost-no-size-state-deltas` | 0.8295 | 0.8832 | 0.8555 |
| `d1-xgboost-no-timing-deltas` | 0.7856 | 0.8241 | 0.8044 |
| `d2-rule-delay` | n/a | n/a | n/a |
| `d2-rule-interval-t` | n/a | n/a | n/a |
| `d2-rule-interval-timestamp` | n/a | n/a | n/a |
| `d2-rule-sqnum-gap` | n/a | n/a | n/a |
| `d2-rule-stnum-gap` | n/a | n/a | n/a |
| `d2-rule-time-since-change` | n/a | n/a | n/a |
| `v2-decision-tree-downsample` | 0.4858 | 0.9585 | 0.6448 |
| `v2-decision-tree-none` | 0.8146 | 0.8421 | 0.8281 |
| `v2-decision-tree-none-cap4m` | 0.8162 | 0.8370 | 0.8265 |
| `v2-logistic-regression-downsample` | 0.4129 | 0.9266 | 0.5713 |
| `v2-logistic-regression-none` | 0.5624 | 0.5953 | 0.5784 |
| `v2-random-forest-downsample` | 0.5716 | 0.8962 | 0.6980 |
| `v2-random-forest-none-cap4m` | 0.8186 | 0.8415 | 0.8299 |
| `v2-xgboost-downsample` | 0.5187 | 0.9583 | 0.6731 |
| `v2-xgboost-none` | 0.8298 | 0.8828 | 0.8555 |
| `d4-xgboost-tuned` | 0.8299 | 0.8826 | 0.8554 |

### `FRG`

63,963 rows across all folds.

| Run | precision | recall | F1 |
|---|---:|---:|---:|
| `d1-xgboost-no-absolute-time` | 0.6504 | 0.6719 | 0.6610 |
| `d1-xgboost-no-counters` | 0.6491 | 0.6732 | 0.6609 |
| `d1-xgboost-no-delta` | 0.0000 | 0.0000 | 0.0000 |
| `d1-xgboost-no-electrical` | 0.6633 | 0.6558 | 0.6595 |
| `d1-xgboost-no-goose-header` | 0.6608 | 0.6551 | 0.6579 |
| `d1-xgboost-no-sequence` | 0.6377 | 0.6176 | 0.6275 |
| `d1-xgboost-no-size-state-deltas` | 0.6616 | 0.6575 | 0.6596 |
| `d1-xgboost-no-timing-deltas` | 0.5095 | 0.5187 | 0.5141 |
| `d2-rule-delay` | 0.0077 | 0.9648 | 0.0153 |
| `d2-rule-interval-t` | 0.0093 | 0.1007 | 0.0170 |
| `d2-rule-interval-timestamp` | 0.3014 | 0.7261 | 0.4260 |
| `d2-rule-sqnum-gap` | 0.0870 | 0.7759 | 0.1565 |
| `d2-rule-stnum-gap` | 0.0078 | 0.0273 | 0.0121 |
| `d2-rule-time-since-change` | 0.0058 | 1.0000 | 0.0115 |
| `v2-decision-tree-downsample` | 0.2645 | 0.7247 | 0.3876 |
| `v2-decision-tree-none` | 0.7058 | 0.6590 | 0.6816 |
| `v2-decision-tree-none-cap4m` | 0.7074 | 0.6532 | 0.6792 |
| `v2-logistic-regression-downsample` | 0.0290 | 0.3769 | 0.0538 |
| `v2-logistic-regression-none` | 0.0000 | 0.0000 | 0.0000 |
| `v2-random-forest-downsample` | 0.2161 | 0.7814 | 0.3385 |
| `v2-random-forest-none-cap4m` | 0.6408 | 0.5954 | 0.6173 |
| `v2-xgboost-downsample` | 0.2733 | 0.7780 | 0.4045 |
| `v2-xgboost-none` | 0.6634 | 0.6529 | 0.6581 |
| `d4-xgboost-tuned` | 0.6630 | 0.6539 | 0.6584 |

### `SAG.PB`

46,959 rows across all folds.

| Run | precision | recall | F1 |
|---|---:|---:|---:|
| `d1-xgboost-no-absolute-time` | 0.8446 | 0.7212 | 0.7780 |
| `d1-xgboost-no-counters` | 0.8408 | 0.7181 | 0.7746 |
| `d1-xgboost-no-delta` | 0.0000 | 0.0000 | 0.0000 |
| `d1-xgboost-no-electrical` | 0.8371 | 0.7281 | 0.7788 |
| `d1-xgboost-no-goose-header` | 0.8408 | 0.7260 | 0.7792 |
| `d1-xgboost-no-sequence` | 0.8333 | 0.7110 | 0.7673 |
| `d1-xgboost-no-size-state-deltas` | 0.8403 | 0.7285 | 0.7804 |
| `d1-xgboost-no-timing-deltas` | 0.8124 | 0.6441 | 0.7185 |
| `d2-rule-delay` | n/a | n/a | n/a |
| `d2-rule-interval-t` | n/a | n/a | n/a |
| `d2-rule-interval-timestamp` | n/a | n/a | n/a |
| `d2-rule-sqnum-gap` | n/a | n/a | n/a |
| `d2-rule-stnum-gap` | n/a | n/a | n/a |
| `d2-rule-time-since-change` | n/a | n/a | n/a |
| `v2-decision-tree-downsample` | 0.2741 | 0.7928 | 0.4074 |
| `v2-decision-tree-none` | 0.8333 | 0.6741 | 0.7453 |
| `v2-decision-tree-none-cap4m` | 0.8272 | 0.6663 | 0.7381 |
| `v2-logistic-regression-downsample` | 0.1124 | 0.7057 | 0.1939 |
| `v2-logistic-regression-none` | 0.6826 | 0.3032 | 0.4199 |
| `v2-random-forest-downsample` | 0.2381 | 0.8345 | 0.3705 |
| `v2-random-forest-none-cap4m` | 0.8128 | 0.7296 | 0.7690 |
| `v2-xgboost-downsample` | 0.2962 | 0.8339 | 0.4372 |
| `v2-xgboost-none` | 0.8407 | 0.7272 | 0.7798 |
| `d4-xgboost-tuned` | 0.8401 | 0.7263 | 0.7791 |

### `SAG.PBM`

54,828 rows across all folds.

| Run | precision | recall | F1 |
|---|---:|---:|---:|
| `d1-xgboost-no-absolute-time` | 0.6956 | 0.2560 | 0.3743 |
| `d1-xgboost-no-counters` | 0.6955 | 0.2652 | 0.3839 |
| `d1-xgboost-no-delta` | 0.0000 | 0.0000 | 0.0000 |
| `d1-xgboost-no-electrical` | 0.6997 | 0.2720 | 0.3917 |
| `d1-xgboost-no-goose-header` | 0.7013 | 0.2721 | 0.3921 |
| `d1-xgboost-no-sequence` | 0.6881 | 0.2710 | 0.3888 |
| `d1-xgboost-no-size-state-deltas` | 0.6999 | 0.2636 | 0.3830 |
| `d1-xgboost-no-timing-deltas` | 0.5931 | 0.2208 | 0.3218 |
| `d2-rule-delay` | n/a | n/a | n/a |
| `d2-rule-interval-t` | n/a | n/a | n/a |
| `d2-rule-interval-timestamp` | n/a | n/a | n/a |
| `d2-rule-sqnum-gap` | n/a | n/a | n/a |
| `d2-rule-stnum-gap` | n/a | n/a | n/a |
| `d2-rule-time-since-change` | n/a | n/a | n/a |
| `v2-decision-tree-downsample` | 0.0431 | 0.8254 | 0.0820 |
| `v2-decision-tree-none` | 0.6562 | 0.1819 | 0.2848 |
| `v2-decision-tree-none-cap4m` | 0.6554 | 0.1865 | 0.2904 |
| `v2-logistic-regression-downsample` | 0.0351 | 0.7529 | 0.0670 |
| `v2-logistic-regression-none` | 0.0369 | 0.0007 | 0.0013 |
| `v2-random-forest-downsample` | 0.0646 | 0.8161 | 0.1196 |
| `v2-random-forest-none-cap4m` | 0.6151 | 0.3965 | 0.4822 |
| `v2-xgboost-downsample` | 0.0547 | 0.8197 | 0.1025 |
| `v2-xgboost-none` | 0.7064 | 0.2721 | 0.3929 |
| `d4-xgboost-tuned` | 0.7022 | 0.2668 | 0.3866 |

### `benign_degradation`

270,642 rows across all folds.

| Run | precision | recall | F1 |
|---|---:|---:|---:|
| `d1-xgboost-no-absolute-time` | 0.9185 | 0.5737 | 0.7063 |
| `d1-xgboost-no-counters` | 0.9096 | 0.5613 | 0.6942 |
| `d1-xgboost-no-delta` | 0.9859 | 0.2261 | 0.3679 |
| `d1-xgboost-no-electrical` | 0.9098 | 0.5628 | 0.6954 |
| `d1-xgboost-no-goose-header` | 0.9149 | 0.5745 | 0.7058 |
| `d1-xgboost-no-sequence` | 0.9140 | 0.5365 | 0.6761 |
| `d1-xgboost-no-size-state-deltas` | 0.9122 | 0.5763 | 0.7063 |
| `d1-xgboost-no-timing-deltas` | 0.8470 | 0.3767 | 0.5215 |
| `d2-rule-delay` | 0.0000 | 0.0000 | 0.0000 |
| `d2-rule-interval-t` | 0.0000 | 0.0000 | 0.0000 |
| `d2-rule-interval-timestamp` | 0.0000 | 0.0000 | 0.0000 |
| `d2-rule-sqnum-gap` | 0.0000 | 0.0000 | 0.0000 |
| `d2-rule-stnum-gap` | 0.0000 | 0.0000 | 0.0000 |
| `d2-rule-time-since-change` | 0.0000 | 0.0000 | 0.0000 |
| `v2-decision-tree-downsample` | 0.1415 | 0.8279 | 0.2417 |
| `v2-decision-tree-none` | 0.9099 | 0.5617 | 0.6946 |
| `v2-decision-tree-none-cap4m` | 0.9100 | 0.5615 | 0.6945 |
| `v2-logistic-regression-downsample` | 0.1043 | 0.4718 | 0.1709 |
| `v2-logistic-regression-none` | 0.5340 | 0.1413 | 0.2234 |
| `v2-random-forest-downsample` | 0.1651 | 0.8479 | 0.2764 |
| `v2-random-forest-none-cap4m` | 0.8818 | 0.6110 | 0.7218 |
| `v2-xgboost-downsample` | 0.1528 | 0.8618 | 0.2595 |
| `v2-xgboost-none` | 0.9134 | 0.5768 | 0.7071 |
| `d4-xgboost-tuned` | 0.9117 | 0.5745 | 0.7048 |

### `normal`

10,570,307 rows across all folds.

| Run | precision | recall | F1 |
|---|---:|---:|---:|
| `d1-xgboost-no-absolute-time` | 0.9863 | 0.9991 | 0.9927 |
| `d1-xgboost-no-counters` | 0.9861 | 0.9990 | 0.9925 |
| `d1-xgboost-no-delta` | 0.9613 | 0.9997 | 0.9801 |
| `d1-xgboost-no-electrical` | 0.9860 | 0.9991 | 0.9925 |
| `d1-xgboost-no-goose-header` | 0.9863 | 0.9992 | 0.9927 |
| `d1-xgboost-no-sequence` | 0.9855 | 0.9995 | 0.9924 |
| `d1-xgboost-no-size-state-deltas` | 0.9863 | 0.9991 | 0.9927 |
| `d1-xgboost-no-timing-deltas` | 0.9809 | 0.9986 | 0.9897 |
| `d2-rule-delay` | 0.9902 | 0.2880 | 0.4462 |
| `d2-rule-interval-t` | 0.9654 | 0.9464 | 0.9558 |
| `d2-rule-interval-timestamp` | 0.9690 | 0.9995 | 0.9840 |
| `d2-rule-sqnum-gap` | 0.9668 | 0.9592 | 0.9630 |
| `d2-rule-stnum-gap` | 0.9645 | 0.9885 | 0.9763 |
| `d2-rule-time-since-change` | 0.0000 | 0.0000 | 0.0000 |
| `v2-decision-tree-downsample` | 0.9997 | 0.7579 | 0.8621 |
| `v2-decision-tree-none` | 0.9843 | 0.9987 | 0.9914 |
| `v2-decision-tree-none-cap4m` | 0.9842 | 0.9987 | 0.9914 |
| `v2-logistic-regression-downsample` | 0.9966 | 0.6992 | 0.8219 |
| `v2-logistic-regression-none` | 0.9676 | 0.9988 | 0.9829 |
| `v2-random-forest-downsample` | 0.9989 | 0.8032 | 0.8904 |
| `v2-random-forest-none-cap4m` | 0.9878 | 0.9981 | 0.9929 |
| `v2-xgboost-downsample` | 0.9996 | 0.7850 | 0.8794 |
| `v2-xgboost-none` | 0.9863 | 0.9992 | 0.9927 |
| `d4-xgboost-tuned` | 0.9862 | 0.9991 | 0.9926 |

`n/a` marks a cell that is **not a result**. A single-threshold rule has one score and cannot name an attack family, so `run_rule_baseline.py` puts that score on one designated class and exactly zero on the other three; those three would come back at the prevalence floor by construction and mean nothing next to a learned run's (`ablations_baselines.md` §16). Only a rule's `ANY_ATTACK` column is a result.

## Confusion matrices

Built from each run's `grouped_predictions.csv`, over the full six-class vocabulary, so ideal `normal` and `benign_degradation` are never folded into one bucket (checklist C.3). Every matrix here also **verifies the table above**: the pooled metrics reconstructed from the per-fold report are checked against `check_prediction_integrity.metrics_from_confusion` on this matrix, and a disagreement beyond 1e-09 is fatal.

### `v2-xgboost-none`

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,561,353 | 7,601 | 65 | 118 | 914 | 256 | 10,570,307 |
| benign_degradation | 92,119 | 156,104 | 1,723 | 16,731 | 2,031 | 1,934 | 270,642 |
| SAG.DB | 4,162 | 66 | 44,828 | 4 | 1,715 | 4 | 50,779 |
| FRG | 12,327 | 5,157 | 338 | 41,761 | 448 | 3,932 | 63,963 |
| SAG.PB | 5,953 | 458 | 6,200 | 124 | 34,148 | 76 | 46,959 |
| SAG.PBM | 31,942 | 1,525 | 868 | 4,213 | 1,361 | 14,919 | 54,828 |

### `d4-xgboost-tuned`

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,560,924 | 7,993 | 48 | 111 | 949 | 282 | 10,570,307 |
| benign_degradation | 92,727 | 155,483 | 1,705 | 16,781 | 2,008 | 1,938 | 270,642 |
| SAG.DB | 4,162 | 60 | 44,816 | 4 | 1,731 | 6 | 50,779 |
| FRG | 12,395 | 5,056 | 338 | 41,827 | 442 | 3,905 | 63,963 |
| SAG.PB | 5,971 | 446 | 6,225 | 138 | 34,107 | 72 | 46,959 |
| SAG.PBM | 32,239 | 1,502 | 869 | 4,229 | 1,363 | 14,626 | 54,828 |

### `d1-xgboost-no-delta`

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,567,455 | 868 | 1,979 | 5 | 0 | 0 | 10,570,307 |
| benign_degradation | 209,443 | 61,197 | 0 | 2 | 0 | 0 | 270,642 |
| SAG.DB | 50,319 | 3 | 457 | 0 | 0 | 0 | 50,779 |
| FRG | 63,904 | 5 | 54 | 0 | 0 | 0 | 63,963 |
| SAG.PB | 46,958 | 0 | 1 | 0 | 0 | 0 | 46,959 |
| SAG.PBM | 54,827 | 0 | 1 | 0 | 0 | 0 | 54,828 |

### `d1-xgboost-no-timing-deltas`

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,555,630 | 3,183 | 2,370 | 5,511 | 1,962 | 1,651 | 10,570,307 |
| benign_degradation | 138,519 | 101,958 | 2,463 | 22,640 | 1,904 | 3,158 | 270,642 |
| SAG.DB | 7,328 | 84 | 41,848 | 0 | 1,519 | 0 | 50,779 |
| FRG | 14,015 | 12,663 | 235 | 33,180 | 432 | 3,438 | 63,963 |
| SAG.PB | 10,234 | 570 | 5,741 | 114 | 30,244 | 56 | 46,959 |
| SAG.PBM | 35,360 | 1,911 | 610 | 3,677 | 1,166 | 12,104 | 54,828 |

### `d2-rule-interval-timestamp`

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 10,564,930 | 0 | 0 | 5,377 | 0 | 0 | 10,570,307 |
| benign_degradation | 217,830 | 0 | 0 | 52,812 | 0 | 0 | 270,642 |
| SAG.DB | 30,304 | 0 | 0 | 20,475 | 0 | 0 | 50,779 |
| FRG | 17,522 | 0 | 0 | 46,441 | 0 | 0 | 63,963 |
| SAG.PB | 25,719 | 0 | 0 | 21,240 | 0 | 0 | 46,959 |
| SAG.PBM | 47,089 | 0 | 0 | 7,739 | 0 | 0 | 54,828 |

## Where the uncertainty lives

| Question | The artifact that answers it |
|---|---|
| Does configuration A rank attacks better than B? | `grouped_pr_curves.py`, paired AP and recall at four alert budgets |
| How much does a per-class number depend on which runs were generated? | `bootstrap_run_intervals.py`, resampling runs |
| Do a run's predictions reconcile with its own report and posteriors? | `check_prediction_integrity.py` |
| When a benign-degradation row is misclassified, what does it become? | `benign_confusion_report.py` |
