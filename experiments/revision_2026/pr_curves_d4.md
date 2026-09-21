# Threshold curves over the grouped folds

- Generated: 2026-09-21 11:45:43 UTC
- Threshold grid: 2,048 points, uniform in the logit over +/-20 (resolves to ~2e-9 from either end)
- Bootstrap: 1,000 replicates, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run)**, as in `bootstrap_run_intervals.py`.

Every number in `grouped_validation_report.json` is `argmax(p)` - one
point on the curves below, at whichever threshold the training prior
put it. AP is the threshold-free summary; the budget tables are the
operating points an alert burden actually allows.

**Read AP against the prevalence column**, not against 1.0: a random
detector scores AP = prevalence, so for `SAG.DB` at 0.219% of the pool
the floor is 0.0022, not 0.

## `v2-xgboost-none`

- model: `xgboost` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.8839 [0.8349, 0.9161] | 192.5x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.6395 [0.5484, 0.7284] | 110.6x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.8551 [0.8247, 0.8797] | 201.3x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.4094 [0.3577, 0.4601] | 82.6x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.8329 [0.8037, 0.8618] | 42.5x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.974038011, 0.975008207, 0.975479967, ... | 0.0106% | 0.0231 [0.0107, 0.0402] | 0.9958 [0.9851, 1.0000] | 1,177 | 1,172 |
| `FRG` | 0.914839128, 0.931419125, 0.938540085, ... | 0.0114% | 0.0196 [0.0043, 0.0406] | 0.9952 [0.9744, 1.0000] | 1,261 | 1,255 |
| `SAG.PB` | 0.999974615, 0.999979525, 0.999981064, ... | 0.0104% | 0.0245 [0.0110, 0.0409] | 1.0000 [1.0000, 1.0000] | 1,151 | 1,151 |
| `SAG.PBM` | 0.991325385, 0.996011222, 0.996450944, ... | 0.0108% | 0.0219 [0.0061, 0.0410] | 0.9992 [0.9956, 1.0000] | 1,199 | 1,198 |
| `ANY_ATTACK` | 0.999990989, 0.999992442, 0.999992872, ... | 0.0117% | 0.0060 [0.0034, 0.0093] | 1.0000 [1.0000, 1.0000] | 1,296 | 1,296 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 6 (100.0%) | 0 (0.0%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 (0.0%) | 1 (100.0%) | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) |
| `ANY_ATTACK` | — | — | — | — | 0 | 0 |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.914839128, 0.917834964, 0.920734542, ... | 0.1047% | 0.2175 [0.1579, 0.2777] | 0.9542 [0.9192, 0.9752] | 11,576 | 11,046 |
| `FRG` | 0.78574453, 0.792250436, 0.795448284 | 0.0995% | 0.1337 [0.0985, 0.1740] | 0.7770 [0.6357, 0.9027] | 11,005 | 8,551 |
| `SAG.PB` | 0.998981219, 0.999093827, 0.999310562, ... | 0.1002% | 0.2356 [0.1685, 0.3057] | 0.9983 [0.9939, 1.0000] | 11,083 | 11,064 |
| `SAG.PBM` | 0.612826417, 0.622057433, 0.635737865, ... | 0.1000% | 0.1702 [0.1509, 0.1915] | 0.8442 [0.7685, 0.9055] | 11,053 | 9,331 |
| `ANY_ATTACK` | 0.999932566, 0.999943441, 0.999946661, ... | 0.1000% | 0.0511 [0.0345, 0.0709] | 1.0000 [1.0000, 1.0000] | 11,056 | 11,056 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 3 (0.6%) | 378 (71.3%) | 10 (1.9%) | 139 (26.2%) | 0 (0.0%) |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 13 (0.5%) | 2,440 (99.4%) | 1 (0.0%) |
| `SAG.PB` | 19 (100.0%) | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `SAG.PBM` | 0 (0.0%) | 1,236 (71.8%) | 0 (0.0%) | — | 441 (25.6%) | 45 (2.6%) |
| `ANY_ATTACK` | — | — | — | — | 0 | 0 |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.000790416934, 0.00108022341, 0.00141963404, ... | 0.9873% | 0.9990 [0.9975, 1.0000] | 0.4647 [0.3824, 0.5401] | 109,170 | 50,730 |
| `FRG` | 0.0571028293, 0.0750912047, 0.0764596808, ... | 0.9885% | 0.8186 [0.8029, 0.8326] | 0.4791 [0.3875, 0.5618] | 109,301 | 52,361 |
| `SAG.PB` | 0.041634622, 0.0432222617, 0.0440376051, ... | 0.9927% | 0.9218 [0.9130, 0.9301] | 0.3944 [0.3209, 0.4595] | 109,770 | 43,288 |
| `SAG.PBM` | 0.0637532549, 0.0673431316, 0.0685808754 | 0.9942% | 0.5105 [0.4702, 0.5465] | 0.2546 [0.2004, 0.3143] | 109,937 | 27,992 |
| `ANY_ATTACK` | 0.810887762, 0.825417645, 0.844260776, ... | 0.9996% | 0.4833 [0.4349, 0.5368] | 0.9467 [0.9254, 0.9658] | 110,535 | 104,647 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 1,297 (2.2%) | 10,275 (17.6%) | 2,768 (4.7%) | 8,259 (14.1%) | 35,841 (61.3%) |
| `FRG` | 793 (1.4%) | — | 867 (1.5%) | 14,060 (24.7%) | 29,899 (52.5%) | 11,321 (19.9%) |
| `SAG.PB` | 35,417 (53.3%) | 1,208 (1.8%) | — | 3,248 (4.9%) | 6,054 (9.1%) | 20,555 (30.9%) |
| `SAG.PBM` | 2,280 (2.8%) | 16,609 (20.3%) | 4,722 (5.8%) | — | 11,436 (14.0%) | 46,898 (57.2%) |
| `ANY_ATTACK` | — | — | — | — | 5,530 (93.9%) | 358 (6.1%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 9.71279386e-07, 1.00998999e-06, 1.05024342e-06, ... | 9.8212% | 1.0000 [1.0000, 1.0000] | 0.0468 [0.0338, 0.0614] | 1,085,975 | 50,779 |
| `FRG` | 4.93288741e-05, 6.61288202e-05, 8.52523622e-05, ... | 10.6054% | 0.9998 [0.9994, 1.0000] | 0.0545 [0.0401, 0.0703] | 1,172,691 | 63,952 |
| `SAG.PB` | 1.44035494e-05, 1.5881853e-05, 1.65148188e-05, ... | 10.0899% | 1.0000 [1.0000, 1.0000] | 0.0421 [0.0337, 0.0501] | 1,115,692 | 46,959 |
| `SAG.PBM` | 0.00054540613, 0.000556162622, 0.000689437526, ... | 10.0364% | 0.9980 [0.9943, 0.9998] | 0.0493 [0.0369, 0.0639] | 1,109,772 | 54,719 |
| `ANY_ATTACK` | 0.0143354395, 0.0148982709, 0.0151877929, ... | 9.9503% | 0.9841 [0.9773, 0.9895] | 0.1937 [0.1718, 0.2183] | 1,100,249 | 213,095 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 52,229 (5.0%) | 38,186 (3.7%) | 40,033 (3.9%) | 84,267 (8.1%) | 820,481 (79.3%) |
| `FRG` | 44,162 (4.0%) | — | 13,746 (1.2%) | 41,074 (3.7%) | 129,527 (11.7%) | 880,230 (79.4%) |
| `SAG.PB` | 50,779 (4.8%) | 21,648 (2.0%) | — | 43,817 (4.1%) | 47,052 (4.4%) | 905,437 (84.7%) |
| `SAG.PBM` | 35,926 (3.4%) | 29,865 (2.8%) | 25,944 (2.5%) | — | 39,294 (3.7%) | 924,024 (87.6%) |
| `ANY_ATTACK` | — | — | — | — | 57,900 (6.5%) | 829,254 (93.5%) |

## `d4-xgboost-tuned`

- model: `xgboost` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.8841 [0.8357, 0.9164] | 192.5x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.6384 [0.5460, 0.7281] | 110.4x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.8548 [0.8244, 0.8795] | 201.3x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.4081 [0.3557, 0.4592] | 82.3x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.8328 [0.8037, 0.8619] | 42.5x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.971986453, 0.972513646, 0.973031193, ... | 0.0108% | 0.0234 [0.0100, 0.0417] | 0.9966 [0.9872, 1.0000] | 1,192 | 1,188 |
| `FRG` | 0.916349232, 0.927578437, 0.936246745, ... | 0.0110% | 0.0189 [0.0043, 0.0390] | 0.9959 [0.9769, 1.0000] | 1,212 | 1,207 |
| `SAG.PB` | 0.999977424, 0.999978709, 0.999982143, ... | 0.0108% | 0.0255 [0.0117, 0.0425] | 1.0000 [1.0000, 1.0000] | 1,197 | 1,197 |
| `SAG.PBM` | 0.990443563, 0.995603628, 0.995688336, ... | 0.0109% | 0.0219 [0.0061, 0.0406] | 0.9992 [0.9956, 1.0000] | 1,200 | 1,199 |
| `ANY_ATTACK` | 0.999991163, 0.999991334, 0.999991502, ... | 0.0099% | 0.0051 [0.0028, 0.0081] | 1.0000 [1.0000, 1.0000] | 1,099 | 1,099 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 (0.0%) | 4 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 5 (100.0%) | 0 (0.0%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 (0.0%) | 1 (100.0%) | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) |
| `ANY_ATTACK` | — | — | — | — | 0 | 0 |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.913304342, 0.916349232, 0.917834964, ... | 0.1037% | 0.2155 [0.1519, 0.2750] | 0.9546 [0.9193, 0.9752] | 11,464 | 10,943 |
| `FRG` | 0.782436471, 0.78574453, 0.792250436, ... | 0.0992% | 0.1325 [0.0963, 0.1742] | 0.7729 [0.6257, 0.9007] | 10,965 | 8,475 |
| `SAG.PB` | 0.999000914, 0.999145378, 0.999268968, ... | 0.1002% | 0.2358 [0.1670, 0.3062] | 0.9995 [0.9984, 1.0000] | 11,079 | 11,074 |
| `SAG.PBM` | 0.612826417, 0.622057433, 0.626640435, ... | 0.0992% | 0.1694 [0.1505, 0.1901] | 0.8463 [0.7696, 0.9073] | 10,973 | 9,286 |
| `ANY_ATTACK` | 0.999936406, 0.999937636, 0.999940026, ... | 0.1008% | 0.0515 [0.0355, 0.0704] | 1.0000 [1.0000, 1.0000] | 11,141 | 11,141 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 5 (1.0%) | 366 (70.2%) | 11 (2.1%) | 139 (26.7%) | 0 (0.0%) |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 11 (0.4%) | 2,477 (99.5%) | 2 (0.1%) |
| `SAG.PB` | 5 (100.0%) | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `SAG.PBM` | 0 (0.0%) | 1,214 (72.0%) | 0 (0.0%) | — | 443 (26.3%) | 30 (1.8%) |
| `ANY_ATTACK` | — | — | — | — | 0 | 0 |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.000942254798, 0.0013130369, 0.00169213884, ... | 0.9878% | 0.9995 [0.9984, 1.0000] | 0.4646 [0.3823, 0.5398] | 109,231 | 50,753 |
| `FRG` | 0.0560597725, 0.0737452658, 0.0764596808, ... | 0.9906% | 0.8188 [0.8031, 0.8323] | 0.4781 [0.3869, 0.5596] | 109,539 | 52,376 |
| `SAG.PB` | 0.0408618674, 0.0424213442, 0.0440376051, ... | 0.9962% | 0.9216 [0.9131, 0.9294] | 0.3929 [0.3192, 0.4585] | 110,151 | 43,277 |
| `SAG.PBM` | 0.0625967842, 0.0673431316, 0.0685808754 | 0.9857% | 0.5093 [0.4691, 0.5453] | 0.2562 [0.2012, 0.3162] | 108,993 | 27,922 |
| `ANY_ATTACK` | 0.810887762, 0.813866131, 0.825417645, ... | 0.9988% | 0.4833 [0.4355, 0.5375] | 0.9477 [0.9261, 0.9668] | 110,438 | 104,659 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 1,306 (2.2%) | 10,278 (17.6%) | 2,760 (4.7%) | 8,371 (14.3%) | 35,763 (61.2%) |
| `FRG` | 771 (1.3%) | — | 890 (1.6%) | 13,958 (24.4%) | 29,981 (52.4%) | 11,563 (20.2%) |
| `SAG.PB` | 36,352 (54.4%) | 1,220 (1.8%) | — | 3,267 (4.9%) | 6,153 (9.2%) | 19,882 (29.7%) |
| `SAG.PBM` | 2,336 (2.9%) | 16,720 (20.6%) | 4,786 (5.9%) | — | 11,422 (14.1%) | 45,807 (56.5%) |
| `ANY_ATTACK` | — | — | — | — | 5,429 (93.9%) | 350 (6.1%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 1.18088786e-06, 1.22795246e-06, 1.25218357e-06, ... | 9.8289% | 1.0000 [1.0000, 1.0000] | 0.0467 [0.0339, 0.0611] | 1,086,829 | 50,779 |
| `FRG` | 5.030223e-05, 5.88132535e-05, 7.15046477e-05, ... | 10.2515% | 0.9999 [0.9995, 1.0000] | 0.0564 [0.0412, 0.0728] | 1,133,561 | 63,954 |
| `SAG.PB` | 1.46877699e-05, 1.5881853e-05, 1.6195244e-05, ... | 9.8916% | 1.0000 [1.0000, 1.0000] | 0.0429 [0.0343, 0.0510] | 1,093,757 | 46,959 |
| `SAG.PBM` | 0.000589720995, 0.000625302983, 0.00076014506, ... | 10.0856% | 0.9981 [0.9945, 0.9998] | 0.0491 [0.0367, 0.0636] | 1,115,215 | 54,722 |
| `ANY_ATTACK` | 0.0146141861, 0.0148982709, 0.0151877929, ... | 9.9830% | 0.9841 [0.9772, 0.9895] | 0.1930 [0.1710, 0.2180] | 1,103,865 | 213,076 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 47,588 (4.6%) | 39,147 (3.8%) | 40,615 (3.9%) | 79,007 (7.6%) | 829,693 (80.1%) |
| `FRG` | 44,755 (4.2%) | — | 13,829 (1.3%) | 41,083 (3.8%) | 129,553 (12.1%) | 840,387 (78.6%) |
| `SAG.PB` | 50,779 (4.9%) | 22,354 (2.1%) | — | 45,045 (4.3%) | 50,221 (4.8%) | 878,399 (83.9%) |
| `SAG.PBM` | 35,998 (3.4%) | 29,926 (2.8%) | 25,518 (2.4%) | — | 38,323 (3.6%) | 930,728 (87.8%) |
| `ANY_ATTACK` | — | — | — | — | 57,980 (6.5%) | 832,809 (93.5%) |

## Paired comparison

Each replicate draws one set of runs and scores **both** configurations
on it, so the run-to-run variation they share cancels instead of being
counted twice. Two marginal intervals overlapping does not mean two
configurations are indistinguishable; this is the table that settles it.
Each configuration keeps its own cross-fold thresholds - the question is
which is better when each is operated properly, not which wins at a
threshold borrowed from the other.

### `v2-xgboost-none` (A) vs `d4-xgboost-tuned` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0002 | [-0.0008, +0.0012] | no - indistinguishable |
| `FRG` | -0.0011 | [-0.0029, +0.0009] | no - indistinguishable |
| `SAG.PB` | -0.0003 | [-0.0009, +0.0004] | no - indistinguishable |
| `SAG.PBM` | -0.0013 | [-0.0023, -0.0003] | **yes** |
| `ANY_ATTACK` | -0.0001 | [-0.0003, +0.0002] | no - indistinguishable |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0003 | [-0.0026, +0.0039] | no |
| `FRG` | -0.0008 | [-0.0024, +0.0007] | no |
| `SAG.PB` | +0.0010 | [-0.0063, +0.0089] | no |
| `SAG.PBM` | +0.0000 | [-0.0008, +0.0009] | no |
| `ANY_ATTACK` | -0.0009 | [-0.0033, +0.0009] | no |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0020 | [-0.0065, +0.0020] | no |
| `FRG` | -0.0012 | [-0.0039, +0.0018] | no |
| `SAG.PB` | +0.0002 | [-0.0075, +0.0081] | no |
| `SAG.PBM` | -0.0008 | [-0.0031, +0.0014] | no |
| `ANY_ATTACK` | +0.0004 | [-0.0038, +0.0043] | no |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0005 | [+0.0000, +0.0015] | no |
| `FRG` | +0.0002 | [-0.0010, +0.0014] | no |
| `SAG.PB` | -0.0002 | [-0.0010, +0.0005] | no |
| `SAG.PBM` | -0.0013 | [-0.0026, -0.0001] | **yes** |
| `ANY_ATTACK` | +0.0001 | [-0.0014, +0.0016] | no |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0000 | [+0.0000, +0.0001] | no |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0001 | [-0.0000, +0.0002] | no |
| `ANY_ATTACK` | -0.0001 | [-0.0008, +0.0007] | no |

## How to read the budget tables

- A threshold is picked on the **other** folds' rows and applied to the
  fold being scored, so no recall here is inflated by having seen the
  rows it is measured on. The per-fold thresholds are listed because
  their spread *is* the calibration stability question.
- `achieved alert rate` can exceed the budget. That means even the
  strictest grid point alerts more often than the budget allows - the
  model cannot be run that quietly, which is a result, not a rounding
  problem.
- `benign_degradation` in the false-alarm table is the card-C confound
  on this axis: a detector whose alarms are mostly benign impairment is
  keying on "traffic looks degraded", not on the attack.
- The split between *ideal* `normal` and `normal` inside a benign run is
  not made here - that rejoin lives in `benign_confusion_report.py`.

