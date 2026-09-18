# Threshold curves over the grouped folds

- Generated: 2026-09-18 17:47:12 UTC
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

## `d2-rule-interval-timestamp`

- model: `rule:interval-timestamp` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.0046 [0.0032, 0.0063] | 1.0x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.1457 [0.1062, 0.1883] | 25.2x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0042 [0.0035, 0.0050] | 1.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0050 [0.0036, 0.0068] | 1.0x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.2411 [0.1895, 0.3052] | 12.3x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.995852907, 0.996381164, 0.996450944, ... | 0.0102% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1,130 | 0 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.995852907, 0.996381164, 0.996450944, ... | 0.0102% | 0.0005 [0.0002, 0.0007] | 0.0885 [0.0367, 0.2346] | 1,130 | 100 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 79 (7.0%) | — | 21 (1.9%) | 0 (0.0%) | 963 (85.2%) | 67 (5.9%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 963 (93.5%) | 67 (6.5%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.960642637, 0.96417488, 0.965500605, ... | 0.1005% | 0.0002 [0.0001, 0.0003] | 0.0012 [0.0004, 0.0026] | 11,109 | 13 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.960642637, 0.96417488, 0.965500605, ... | 0.1005% | 0.0090 [0.0057, 0.0126] | 0.1755 [0.0995, 0.3212] | 11,109 | 1,950 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 1,426 (12.9%) | — | 511 (4.6%) | 0 (0.0%) | 8,523 (76.8%) | 636 (5.7%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 8,523 (93.1%) | 636 (6.9%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.612826417, 0.626640435, 0.649202358, ... | 0.9988% | 0.3615 [0.3362, 0.3841] | 0.2093 [0.1502, 0.2739] | 110,438 | 23,120 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.612826417, 0.626640435, 0.649202358, ... | 0.9988% | 0.2980 [0.2520, 0.3453] | 0.5842 [0.4976, 0.6748] | 110,438 | 64,516 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 18,255 (20.9%) | — | 20,275 (23.2%) | 2,866 (3.3%) | 42,513 (48.7%) | 3,409 (3.9%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 42,513 (92.6%) | 3,409 (7.4%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.45857102 | 9.5560% | 0.7310 [0.7248, 0.7366] | 0.0443 [0.0310, 0.0599] | 1,056,649 | 46,758 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.45857102 | 9.5560% | 0.4466 [0.3946, 0.4975] | 0.0915 [0.0754, 0.1106] | 1,056,649 | 96,704 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 20,686 (2.0%) | — | 21,330 (2.1%) | 7,930 (0.8%) | 85,012 (8.4%) | 874,933 (86.6%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 85,012 (8.9%) | 874,933 (91.1%) |

## `d2-rule-stnum-gap`

- model: `rule:stnum-gap` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.0046 [0.0032, 0.0063] | 1.0x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.0066 [0.0046, 0.0092] | 1.1x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0042 [0.0035, 0.0050] | 1.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0050 [0.0036, 0.0068] | 1.0x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.2192 [0.1754, 0.2712] | 11.2x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.998981219, 0.999161902 | 0.0119% | 0.0000 [0.0000, 0.0001] | 0.0008 [0.0000, 0.0031] | 1,315 | 1 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.998981219, 0.999161902 | 0.0119% | 0.0050 [0.0029, 0.0077] | 0.8266 [0.6842, 0.9352] | 1,315 | 1,087 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 923 (70.2%) | — | 163 (12.4%) | 0 (0.0%) | 108 (8.2%) | 120 (9.1%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 108 (47.4%) | 120 (52.6%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.981594903, 0.981944631 | 0.5025% | 0.0068 [0.0033, 0.0107] | 0.0078 [0.0032, 0.0144] | 55,566 | 434 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.981594903, 0.981944631 | 0.5025% | 0.1216 [0.0859, 0.1621] | 0.4739 [0.3630, 0.5923] | 55,566 | 26,332 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 16,339 (29.6%) | — | 8,754 (15.9%) | 805 (1.5%) | 3,600 (6.5%) | 25,634 (46.5%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 3,600 (12.3%) | 25,634 (87.7%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.98012832, 0.980505362, 0.981594903 | 0.8073% | 0.0137 [0.0091, 0.0181] | 0.0098 [0.0055, 0.0150] | 89,265 | 874 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.98012832, 0.980505362, 0.981594903 | 0.8073% | 0.1880 [0.1442, 0.2345] | 0.4560 [0.3695, 0.5520] | 89,265 | 40,705 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 25,562 (28.9%) | — | 12,601 (14.3%) | 1,668 (1.9%) | 6,542 (7.4%) | 42,018 (47.5%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 6,542 (13.5%) | 42,018 (86.5%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.502442579 | 2.0238% | 0.0273 [0.0256, 0.0289] | 0.0078 [0.0053, 0.0106] | 223,782 | 1,744 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.502442579 | 2.0238% | 0.4081 [0.3508, 0.4695] | 0.3949 [0.3440, 0.4468] | 223,782 | 88,374 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 49,500 (22.3%) | — | 32,270 (14.5%) | 4,860 (2.2%) | 13,695 (6.2%) | 121,713 (54.8%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 13,695 (10.1%) | 121,713 (89.9%) |

## `d2-rule-interval-t`

- model: `rule:interval-t` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.0046 [0.0032, 0.0063] | 1.0x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.0066 [0.0045, 0.0092] | 1.1x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0042 [0.0035, 0.0050] | 1.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0050 [0.0036, 0.0068] | 1.0x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.1010 [0.0822, 0.1237] | 5.2x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.999193998, 0.999209583, 0.999224867, ... | 0.0099% | 0.0001 [0.0001, 0.0002] | 0.0082 [0.0031, 0.0144] | 1,093 | 9 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.999193998, 0.999209583, 0.999224867, ... | 0.0099% | 0.0007 [0.0006, 0.0009] | 0.1436 [0.1072, 0.1873] | 1,093 | 157 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 88 (8.1%) | — | 50 (4.6%) | 10 (0.9%) | 37 (3.4%) | 899 (82.9%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 37 (4.0%) | 899 (96.0%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.992126484 | 0.0994% | 0.0016 [0.0013, 0.0019] | 0.0091 [0.0059, 0.0133] | 10,993 | 100 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.992126484 | 0.0994% | 0.0075 [0.0066, 0.0085] | 0.1472 [0.1221, 0.1793] | 10,993 | 1,618 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 759 (7.0%) | — | 560 (5.1%) | 199 (1.8%) | 437 (4.0%) | 8,938 (82.1%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 437 (4.7%) | 8,938 (95.3%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.920734542, 0.922149003 | 0.9958% | 0.0183 [0.0173, 0.0193] | 0.0106 [0.0073, 0.0149] | 110,112 | 1,172 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.920734542, 0.922149003 | 0.9958% | 0.0886 [0.0779, 0.1005] | 0.1742 [0.1446, 0.2097] | 110,112 | 19,179 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 10,284 (9.4%) | — | 5,764 (5.3%) | 1,959 (1.8%) | 4,583 (4.2%) | 86,350 (79.3%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 4,583 (5.0%) | 86,350 (95.0%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.482908469 | 9.6423% | 0.1037 [0.1011, 0.1064] | 0.0062 [0.0042, 0.0090] | 1,066,194 | 6,635 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.482908469 | 9.6423% | 0.5237 [0.4693, 0.5817] | 0.1064 [0.0871, 0.1331] | 1,066,194 | 113,397 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 49,490 (4.7%) | — | 39,805 (3.8%) | 17,467 (1.6%) | 27,469 (2.6%) | 925,328 (87.3%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 27,469 (2.9%) | 925,328 (97.1%) |

## `d2-rule-sqnum-gap`

- model: `rule:sqnum-gap` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.0046 [0.0032, 0.0063] | 1.0x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.0523 [0.0371, 0.0708] | 9.0x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0042 [0.0035, 0.0050] | 1.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0050 [0.0036, 0.0068] | 1.0x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.0451 [0.0338, 0.0592] | 2.3x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.999193998, 0.999224867 | 0.0097% | 0.0001 [0.0000, 0.0002] | 0.0056 [0.0016, 0.0110] | 1,072 | 6 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.999193998, 0.999224867 | 0.0097% | 0.0000 [0.0000, 0.0001] | 0.0056 [0.0016, 0.0110] | 1,072 | 6 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 17 (1.6%) | 1,049 (98.4%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 17 (1.6%) | 1,049 (98.4%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.990443563, 0.990626758, 0.990806473 | 0.1000% | 0.0010 [0.0008, 0.0011] | 0.0056 [0.0035, 0.0083] | 11,059 | 62 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.990443563, 0.990626758, 0.990806473 | 0.1000% | 0.0003 [0.0002, 0.0004] | 0.0056 [0.0035, 0.0083] | 11,059 | 62 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 232 (2.1%) | 10,765 (97.9%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 232 (2.1%) | 10,765 (97.9%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.906912174, 0.910159483 | 0.9654% | 0.0100 [0.0091, 0.0109] | 0.0060 [0.0041, 0.0083] | 106,749 | 638 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.906912174, 0.910159483 | 0.9654% | 0.0029 [0.0022, 0.0037] | 0.0060 [0.0041, 0.0083] | 106,749 | 638 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 2,663 (2.5%) | 103,448 (97.5%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 2,663 (2.5%) | 103,448 (97.5%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.502442579 | 5.1571% | 0.7759 [0.7716, 0.7799] | 0.0870 [0.0610, 0.1162] | 570,240 | 49,629 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.502442579 | 5.1571% | 0.3564 [0.3072, 0.4045] | 0.1353 [0.1079, 0.1671] | 570,240 | 77,165 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 1,529 (0.3%) | — | 9,815 (1.9%) | 16,192 (3.1%) | 61,797 (11.9%) | 431,278 (82.8%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 61,797 (12.5%) | 431,278 (87.5%) |

## `d2-rule-delay`

- model: `rule:delay` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.0046 [0.0032, 0.0063] | 1.0x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.0052 [0.0036, 0.0072] | 0.9x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0042 [0.0035, 0.0050] | 1.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0050 [0.0036, 0.0068] | 1.0x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.0175 [0.0147, 0.0211] | 0.9x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.999929879, 0.999933871, 0.999937636, ... | 0.0101% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1,114 | 0 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.999929879, 0.999933871, 0.999937636, ... | 0.0101% | 0.0000 [0.0000, 0.0000] | 0.0027 [0.0000, 0.0057] | 1,114 | 3 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 3 (0.3%) | 0 (0.0%) | 0 (0.0%) | 1,111 (99.7%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 0 (0.0%) | 1,111 (100.0%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.999283104, 0.99933697, 0.99938679 | 0.1012% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 11,189 | 0 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.999283104, 0.99933697, 0.99938679 | 0.1012% | 0.0001 [0.0000, 0.0002] | 0.0017 [0.0007, 0.0028] | 11,189 | 19 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 19 (0.2%) | 0 (0.0%) | 0 (0.0%) | 11,170 (99.8%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 0 (0.0%) | 11,170 (100.0%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.993639671, 0.993761983, 0.993881957 | 1.0456% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 115,615 | 0 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.993639671, 0.993761983, 0.993881957 | 1.0456% | 0.0011 [0.0005, 0.0018] | 0.0020 [0.0012, 0.0029] | 115,615 | 236 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 236 (0.2%) | 0 (0.0%) | 0 (0.0%) | 115,379 (99.8%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 0 (0.0%) | 115,379 (100.0%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.928880205, 0.930160335, 0.931419125, ... | 10.0992% | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1,116,717 | 0 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.928880205, 0.930160335, 0.931419125, ... | 10.0992% | 0.0134 [0.0083, 0.0198] | 0.0026 [0.0018, 0.0036] | 1,116,717 | 2,909 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 2,262 (0.2%) | 647 (0.1%) | 0 (0.0%) | 1,113,808 (99.7%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 0 (0.0%) | 1,113,808 (100.0%) |

## `d2-rule-time-since-change`

- model: `rule:time-since-change` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.0046 [0.0032, 0.0063] | 1.0x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.0055 [0.0038, 0.0076] | 0.9x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0042 [0.0035, 0.0050] | 1.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0050 [0.0036, 0.0068] | 1.0x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.0127 [0.0105, 0.0153] | 0.6x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.999945608, 0.999947693, 0.999950671, ... | 0.0106% | 0.0000 [0.0000, 0.0001] | 0.0017 [0.0000, 0.0059] | 1,175 | 2 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.999945608, 0.999947693, 0.999950671, ... | 0.0106% | 0.0000 [0.0000, 0.0000] | 0.0017 [0.0000, 0.0059] | 1,175 | 2 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1,173 (100.0%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 0 (0.0%) | 1,173 (100.0%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.999465142, 0.999475487, 0.999514904, ... | 0.1006% | 0.0004 [0.0001, 0.0008] | 0.0024 [0.0008, 0.0049] | 11,120 | 27 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.999465142, 0.999475487, 0.999514904, ... | 0.1006% | 0.0001 [0.0000, 0.0002] | 0.0024 [0.0008, 0.0049] | 11,120 | 27 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 52 (0.5%) | 11,041 (99.5%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 52 (0.5%) | 11,041 (99.5%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.994962445, 0.995059445, 0.995154587 | 0.9946% | 0.0088 [0.0071, 0.0104] | 0.0051 [0.0034, 0.0074] | 109,977 | 560 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.994962445, 0.995059445, 0.995154587 | 0.9946% | 0.0026 [0.0019, 0.0034] | 0.0051 [0.0034, 0.0074] | 109,977 | 560 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 2,082 (1.9%) | 107,335 (98.1%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 2,082 (1.9%) | 107,335 (98.1%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `FRG` | 0.949831111, 0.950754126 | 9.8860% | 0.0889 [0.0837, 0.0945] | 0.0052 [0.0036, 0.0073] | 1,093,139 | 5,686 |
| `SAG.PB` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `SAG.PBM` | 2.06115362e-09 | 0.0000% | 0.0000 [0.0000, 0.0000] | n/a | 0 | 0 |
| `ANY_ATTACK` | 0.949831111, 0.950754126 | 9.8860% | 0.0263 [0.0198, 0.0329] | 0.0052 [0.0036, 0.0073] | 1,093,139 | 5,686 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 | 0 | 0 | 0 | 0 |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 23,987 (2.2%) | 1,063,466 (97.8%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 23,987 (2.2%) | 1,063,466 (97.8%) |

## Paired comparison

Each replicate draws one set of runs and scores **both** configurations
on it, so the run-to-run variation they share cancels instead of being
counted twice. Two marginal intervals overlapping does not mean two
configurations are indistinguishable; this is the table that settles it.
Each configuration keeps its own cross-fold thresholds - the question is
which is better when each is operated properly, not which wins at a
threshold borrowed from the other.

### `v2-xgboost-none` (A) vs `d2-rule-interval-timestamp` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8793 | [-0.9112, -0.8318] | **yes** |
| `FRG` | -0.4938 | [-0.5672, -0.4174] | **yes** |
| `SAG.PB` | -0.8508 | [-0.8753, -0.8209] | **yes** |
| `SAG.PBM` | -0.4044 | [-0.4539, -0.3533] | **yes** |
| `ANY_ATTACK` | -0.5918 | [-0.6385, -0.5392] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0231 | [-0.0402, -0.0107] | **yes** |
| `FRG` | -0.0196 | [-0.0406, -0.0043] | **yes** |
| `SAG.PB` | -0.0245 | [-0.0409, -0.0110] | **yes** |
| `SAG.PBM` | -0.0219 | [-0.0410, -0.0061] | **yes** |
| `ANY_ATTACK` | -0.0055 | [-0.0087, -0.0029] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2175 | [-0.2777, -0.1579] | **yes** |
| `FRG` | -0.1335 | [-0.1738, -0.0983] | **yes** |
| `SAG.PB` | -0.2356 | [-0.3057, -0.1685] | **yes** |
| `SAG.PBM` | -0.1702 | [-0.1915, -0.1509] | **yes** |
| `ANY_ATTACK` | -0.0421 | [-0.0616, -0.0267] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9990 | [-1.0000, -0.9975] | **yes** |
| `FRG` | -0.4572 | [-0.4822, -0.4342] | **yes** |
| `SAG.PB` | -0.9218 | [-0.9301, -0.9130] | **yes** |
| `SAG.PBM` | -0.5105 | [-0.5465, -0.4702] | **yes** |
| `ANY_ATTACK` | -0.1853 | [-0.2436, -0.1332] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.2688 | [-0.2749, -0.2633] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9980 | [-0.9998, -0.9943] | **yes** |
| `ANY_ATTACK` | -0.5375 | [-0.5904, -0.4864] | **yes** |

### `v2-xgboost-none` (A) vs `d2-rule-stnum-gap` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8793 | [-0.9112, -0.8318] | **yes** |
| `FRG` | -0.6329 | [-0.7209, -0.5425] | **yes** |
| `SAG.PB` | -0.8508 | [-0.8753, -0.8209] | **yes** |
| `SAG.PBM` | -0.4044 | [-0.4539, -0.3533] | **yes** |
| `ANY_ATTACK` | -0.6137 | [-0.6509, -0.5708] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0231 | [-0.0402, -0.0107] | **yes** |
| `FRG` | -0.0196 | [-0.0405, -0.0043] | **yes** |
| `SAG.PB` | -0.0245 | [-0.0409, -0.0110] | **yes** |
| `SAG.PBM` | -0.0219 | [-0.0410, -0.0061] | **yes** |
| `ANY_ATTACK` | -0.0010 | [-0.0052, +0.0029] | no |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2175 | [-0.2777, -0.1579] | **yes** |
| `FRG` | -0.1269 | [-0.1656, -0.0912] | **yes** |
| `SAG.PB` | -0.2356 | [-0.3057, -0.1685] | **yes** |
| `SAG.PBM` | -0.1702 | [-0.1915, -0.1509] | **yes** |
| `ANY_ATTACK` | +0.0705 | [+0.0313, +0.1106] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9990 | [-1.0000, -0.9975] | **yes** |
| `FRG` | -0.8049 | [-0.8200, -0.7873] | **yes** |
| `SAG.PB` | -0.9218 | [-0.9301, -0.9130] | **yes** |
| `SAG.PBM` | -0.5105 | [-0.5465, -0.4702] | **yes** |
| `ANY_ATTACK` | -0.2953 | [-0.3417, -0.2504] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.9726 | [-0.9743, -0.9709] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9980 | [-0.9998, -0.9943] | **yes** |
| `ANY_ATTACK` | -0.5760 | [-0.6348, -0.5168] | **yes** |

### `v2-xgboost-none` (A) vs `d2-rule-interval-t` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8793 | [-0.9112, -0.8318] | **yes** |
| `FRG` | -0.6329 | [-0.7209, -0.5425] | **yes** |
| `SAG.PB` | -0.8508 | [-0.8753, -0.8209] | **yes** |
| `SAG.PBM` | -0.4044 | [-0.4539, -0.3533] | **yes** |
| `ANY_ATTACK` | -0.7319 | [-0.7588, -0.7035] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0231 | [-0.0402, -0.0107] | **yes** |
| `FRG` | -0.0195 | [-0.0404, -0.0042] | **yes** |
| `SAG.PB` | -0.0245 | [-0.0409, -0.0110] | **yes** |
| `SAG.PBM` | -0.0219 | [-0.0410, -0.0061] | **yes** |
| `ANY_ATTACK` | -0.0053 | [-0.0085, -0.0027] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2175 | [-0.2777, -0.1579] | **yes** |
| `FRG` | -0.1321 | [-0.1722, -0.0967] | **yes** |
| `SAG.PB` | -0.2356 | [-0.3057, -0.1685] | **yes** |
| `SAG.PBM` | -0.1702 | [-0.1915, -0.1509] | **yes** |
| `ANY_ATTACK` | -0.0436 | [-0.0629, -0.0276] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9990 | [-1.0000, -0.9975] | **yes** |
| `FRG` | -0.8003 | [-0.8137, -0.7848] | **yes** |
| `SAG.PB` | -0.9218 | [-0.9301, -0.9130] | **yes** |
| `SAG.PBM` | -0.5105 | [-0.5465, -0.4702] | **yes** |
| `ANY_ATTACK` | -0.3947 | [-0.4386, -0.3557] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.8961 | [-0.8989, -0.8935] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9980 | [-0.9998, -0.9943] | **yes** |
| `ANY_ATTACK` | -0.4604 | [-0.5161, -0.4031] | **yes** |

### `v2-xgboost-none` (A) vs `d2-rule-sqnum-gap` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8793 | [-0.9112, -0.8318] | **yes** |
| `FRG` | -0.5872 | [-0.6711, -0.4995] | **yes** |
| `SAG.PB` | -0.8508 | [-0.8753, -0.8209] | **yes** |
| `SAG.PBM` | -0.4044 | [-0.4539, -0.3533] | **yes** |
| `ANY_ATTACK` | -0.7878 | [-0.8186, -0.7557] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0231 | [-0.0402, -0.0107] | **yes** |
| `FRG` | -0.0195 | [-0.0405, -0.0042] | **yes** |
| `SAG.PB` | -0.0245 | [-0.0409, -0.0110] | **yes** |
| `SAG.PBM` | -0.0219 | [-0.0410, -0.0061] | **yes** |
| `ANY_ATTACK` | -0.0060 | [-0.0092, -0.0033] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2175 | [-0.2777, -0.1579] | **yes** |
| `FRG` | -0.1327 | [-0.1729, -0.0975] | **yes** |
| `SAG.PB` | -0.2356 | [-0.3057, -0.1685] | **yes** |
| `SAG.PBM` | -0.1702 | [-0.1915, -0.1509] | **yes** |
| `ANY_ATTACK` | -0.0508 | [-0.0707, -0.0342] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9990 | [-1.0000, -0.9975] | **yes** |
| `FRG` | -0.8086 | [-0.8227, -0.7926] | **yes** |
| `SAG.PB` | -0.9218 | [-0.9301, -0.9130] | **yes** |
| `SAG.PBM` | -0.5105 | [-0.5465, -0.4702] | **yes** |
| `ANY_ATTACK` | -0.4803 | [-0.5343, -0.4315] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.2239 | [-0.2283, -0.2199] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9980 | [-0.9998, -0.9943] | **yes** |
| `ANY_ATTACK` | -0.6278 | [-0.6769, -0.5795] | **yes** |

### `v2-xgboost-none` (A) vs `d2-rule-delay` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8793 | [-0.9112, -0.8318] | **yes** |
| `FRG` | -0.6343 | [-0.7223, -0.5438] | **yes** |
| `SAG.PB` | -0.8508 | [-0.8753, -0.8209] | **yes** |
| `SAG.PBM` | -0.4044 | [-0.4539, -0.3533] | **yes** |
| `ANY_ATTACK` | -0.8153 | [-0.8439, -0.7867] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0231 | [-0.0402, -0.0107] | **yes** |
| `FRG` | -0.0196 | [-0.0406, -0.0043] | **yes** |
| `SAG.PB` | -0.0245 | [-0.0409, -0.0110] | **yes** |
| `SAG.PBM` | -0.0219 | [-0.0410, -0.0061] | **yes** |
| `ANY_ATTACK` | -0.0060 | [-0.0093, -0.0034] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2175 | [-0.2777, -0.1579] | **yes** |
| `FRG` | -0.1337 | [-0.1740, -0.0985] | **yes** |
| `SAG.PB` | -0.2356 | [-0.3057, -0.1685] | **yes** |
| `SAG.PBM` | -0.1702 | [-0.1915, -0.1509] | **yes** |
| `ANY_ATTACK` | -0.0510 | [-0.0708, -0.0344] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9990 | [-1.0000, -0.9975] | **yes** |
| `FRG` | -0.8186 | [-0.8326, -0.8029] | **yes** |
| `SAG.PB` | -0.9218 | [-0.9301, -0.9130] | **yes** |
| `SAG.PBM` | -0.5105 | [-0.5465, -0.4702] | **yes** |
| `ANY_ATTACK` | -0.4822 | [-0.5354, -0.4342] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.9998 | [-1.0000, -0.9994] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9980 | [-0.9998, -0.9943] | **yes** |
| `ANY_ATTACK` | -0.9707 | [-0.9800, -0.9595] | **yes** |

### `v2-xgboost-none` (A) vs `d2-rule-time-since-change` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8793 | [-0.9112, -0.8318] | **yes** |
| `FRG` | -0.6340 | [-0.7220, -0.5437] | **yes** |
| `SAG.PB` | -0.8508 | [-0.8753, -0.8209] | **yes** |
| `SAG.PBM` | -0.4044 | [-0.4539, -0.3533] | **yes** |
| `ANY_ATTACK` | -0.8202 | [-0.8489, -0.7921] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0231 | [-0.0402, -0.0107] | **yes** |
| `FRG` | -0.0196 | [-0.0404, -0.0043] | **yes** |
| `SAG.PB` | -0.0245 | [-0.0409, -0.0110] | **yes** |
| `SAG.PBM` | -0.0219 | [-0.0410, -0.0061] | **yes** |
| `ANY_ATTACK` | -0.0060 | [-0.0093, -0.0033] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2175 | [-0.2777, -0.1579] | **yes** |
| `FRG` | -0.1333 | [-0.1733, -0.0979] | **yes** |
| `SAG.PB` | -0.2356 | [-0.3057, -0.1685] | **yes** |
| `SAG.PBM` | -0.1702 | [-0.1915, -0.1509] | **yes** |
| `ANY_ATTACK` | -0.0509 | [-0.0708, -0.0344] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9990 | [-1.0000, -0.9975] | **yes** |
| `FRG` | -0.8099 | [-0.8235, -0.7940] | **yes** |
| `SAG.PB` | -0.9218 | [-0.9301, -0.9130] | **yes** |
| `SAG.PBM` | -0.5105 | [-0.5465, -0.4702] | **yes** |
| `ANY_ATTACK` | -0.4807 | [-0.5345, -0.4318] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.9109 | [-0.9162, -0.9053] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9980 | [-0.9998, -0.9943] | **yes** |
| `ANY_ATTACK` | -0.9579 | [-0.9660, -0.9490] | **yes** |

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

