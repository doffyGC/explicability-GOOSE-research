# Threshold curves over the grouped folds

- Generated: 2026-09-23 16:50:52 UTC
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

## `f10-xgboost-no-abs-no-counters`

- model: `xgboost` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.8444 [0.7817, 0.8882] | 183.9x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.5914 [0.4927, 0.6887] | 102.2x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.8285 [0.7930, 0.8583] | 195.1x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.3716 [0.3217, 0.4230] | 74.9x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.8243 [0.7941, 0.8540] | 42.1x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.947935717, 0.950754126, 0.952552052 | 0.0109% | 0.0214 [0.0087, 0.0370] | 0.9036 [0.7457, 0.9657] | 1,203 | 1,087 |
| `FRG` | 0.825417645, 0.828215657, 0.83370486 | 0.0102% | 0.0108 [0.0072, 0.0153] | 0.6105 [0.3923, 0.8476] | 1,127 | 688 |
| `SAG.PB` | 0.999866379, 0.999876424, 0.999890094, ... | 0.0103% | 0.0243 [0.0151, 0.0351] | 1.0000 [1.0000, 1.0000] | 1,142 | 1,142 |
| `SAG.PBM` | 0.966145605, 0.972513646, 0.975479967 | 0.0105% | 0.0211 [0.0124, 0.0307] | 1.0000 [1.0000, 1.0000] | 1,156 | 1,156 |
| `ANY_ATTACK` | 0.999973604, 0.999975107, 0.999975588, ... | 0.0102% | 0.0052 [0.0032, 0.0078] | 1.0000 [1.0000, 1.0000] | 1,130 | 1,130 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 0 (0.0%) | 96 (82.8%) | 0 (0.0%) | 20 (17.2%) | 0 (0.0%) |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 1 (0.2%) | 438 (99.8%) | 0 (0.0%) |
| `SAG.PB` | 0 | 0 | — | 0 | 0 | 0 |
| `SAG.PBM` | 0 | 0 | 0 | — | 0 | 0 |
| `ANY_ATTACK` | — | — | — | — | 0 | 0 |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.874824741, 0.881104483, 0.896530687, ... | 0.1043% | 0.2095 [0.1442, 0.2744] | 0.9227 [0.8652, 0.9584] | 11,528 | 10,637 |
| `FRG` | 0.76534635, 0.77229215, 0.775710239, ... | 0.0981% | 0.1168 [0.0919, 0.1443] | 0.6887 [0.5405, 0.8411] | 10,846 | 7,470 |
| `SAG.PB` | 0.988179035, 0.991491821, 0.992425972, ... | 0.1027% | 0.2418 [0.1829, 0.3083] | 0.9999 [0.9997, 1.0000] | 11,354 | 11,353 |
| `SAG.PBM` | 0.560763164, 0.579914656, 0.594125224, ... | 0.0998% | 0.1608 [0.1468, 0.1754] | 0.7989 [0.7157, 0.8669] | 11,032 | 8,814 |
| `ANY_ATTACK` | 0.996088107, 0.996163516, 0.996381164, ... | 0.0968% | 0.0494 [0.0321, 0.0702] | 1.0000 [1.0000, 1.0000] | 10,704 | 10,704 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 6 (0.7%) | 654 (73.4%) | 28 (3.1%) | 203 (22.8%) | 0 (0.0%) |
| `FRG` | 0 (0.0%) | — | 0 (0.0%) | 11 (0.3%) | 3,365 (99.7%) | 0 (0.0%) |
| `SAG.PB` | 0 (0.0%) | 0 (0.0%) | — | 0 (0.0%) | 0 (0.0%) | 1 (100.0%) |
| `SAG.PBM` | 0 (0.0%) | 1,556 (70.2%) | 0 (0.0%) | — | 628 (28.3%) | 34 (1.5%) |
| `ANY_ATTACK` | — | — | — | — | 0 | 0 |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.0111554207, 0.0332210387, 0.0408618674, ... | 1.0108% | 0.9981 [0.9967, 0.9993] | 0.4535 [0.3711, 0.5313] | 111,765 | 50,685 |
| `FRG` | 0.0560597725, 0.0764596808, 0.0778509971, ... | 1.0008% | 0.8128 [0.8027, 0.8215] | 0.4698 [0.3772, 0.5561] | 110,662 | 51,989 |
| `SAG.PB` | 0.0393573633, 0.0401028552, 0.041634622, ... | 1.0044% | 0.9201 [0.9120, 0.9272] | 0.3890 [0.3159, 0.4559] | 111,062 | 43,208 |
| `SAG.PBM` | 0.0649296114, 0.0711197951 | 0.9581% | 0.5010 [0.4588, 0.5380] | 0.2593 [0.2031, 0.3201] | 105,940 | 27,470 |
| `ANY_ATTACK` | 0.795448284, 0.798609422, 0.807872986, ... | 0.9977% | 0.4804 [0.4307, 0.5331] | 0.9429 [0.9217, 0.9622] | 110,315 | 104,012 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 911 (1.5%) | 13,318 (21.8%) | 2,440 (4.0%) | 4,068 (6.7%) | 40,343 (66.0%) |
| `FRG` | 700 (1.2%) | — | 947 (1.6%) | 15,447 (26.3%) | 31,574 (53.8%) | 10,005 (17.1%) |
| `SAG.PB` | 39,214 (57.8%) | 1,251 (1.8%) | — | 3,203 (4.7%) | 6,798 (10.0%) | 17,388 (25.6%) |
| `SAG.PBM` | 2,210 (2.8%) | 17,195 (21.9%) | 4,116 (5.2%) | — | 11,651 (14.8%) | 43,298 (55.2%) |
| `ANY_ATTACK` | — | — | — | — | 6,038 (95.8%) | 265 (4.2%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 7.53390802e-07, 7.68257423e-07, 8.14640728e-07, ... | 10.0270% | 1.0000 [1.0000, 1.0000] | 0.0458 [0.0330, 0.0608] | 1,108,728 | 50,779 |
| `FRG` | 0.000240116178, 0.00043145155, 0.000556162622, ... | 9.9691% | 0.9999 [0.9998, 1.0000] | 0.0580 [0.0423, 0.0761] | 1,102,328 | 63,957 |
| `SAG.PB` | 1.68406996e-05, 1.71730108e-05, 1.85691348e-05, ... | 10.0180% | 1.0000 [1.0000, 1.0000] | 0.0424 [0.0340, 0.0503] | 1,107,742 | 46,959 |
| `SAG.PBM` | 0.00270191373, 0.0027550836, 0.00383648377, ... | 9.9529% | 0.9889 [0.9787, 0.9966] | 0.0493 [0.0364, 0.0655] | 1,100,542 | 54,217 |
| `ANY_ATTACK` | 0.0143354395, 0.0148982709, 0.0151877929, ... | 9.9630% | 0.9825 [0.9752, 0.9886] | 0.1931 [0.1707, 0.2180] | 1,101,659 | 212,742 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 33,864 (3.2%) | 45,004 (4.3%) | 40,709 (3.8%) | 63,231 (6.0%) | 875,141 (82.7%) |
| `FRG` | 35,485 (3.4%) | — | 21,041 (2.0%) | 48,070 (4.6%) | 81,065 (7.8%) | 852,710 (82.1%) |
| `SAG.PB` | 50,779 (4.8%) | 23,187 (2.2%) | — | 40,573 (3.8%) | 55,030 (5.2%) | 891,214 (84.0%) |
| `SAG.PBM` | 26,988 (2.6%) | 29,338 (2.8%) | 21,624 (2.1%) | — | 28,864 (2.8%) | 939,511 (89.8%) |
| `ANY_ATTACK` | — | — | — | — | 60,187 (6.8%) | 828,730 (93.2%) |

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

### `f10-xgboost-no-abs-no-counters` (A) vs `d2-rule-interval-timestamp` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8398 | [-0.8826, -0.7784] | **yes** |
| `FRG` | -0.4456 | [-0.5269, -0.3681] | **yes** |
| `SAG.PB` | -0.8242 | [-0.8538, -0.7889] | **yes** |
| `SAG.PBM` | -0.3667 | [-0.4169, -0.3173] | **yes** |
| `ANY_ATTACK` | -0.5832 | [-0.6293, -0.5303] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0214 | [-0.0370, -0.0087] | **yes** |
| `FRG` | -0.0108 | [-0.0153, -0.0072] | **yes** |
| `SAG.PB` | -0.0243 | [-0.0351, -0.0151] | **yes** |
| `SAG.PBM` | -0.0211 | [-0.0307, -0.0124] | **yes** |
| `ANY_ATTACK` | -0.0048 | [-0.0073, -0.0026] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2095 | [-0.2744, -0.1442] | **yes** |
| `FRG` | -0.1166 | [-0.1441, -0.0916] | **yes** |
| `SAG.PB` | -0.2418 | [-0.3083, -0.1829] | **yes** |
| `SAG.PBM` | -0.1608 | [-0.1754, -0.1468] | **yes** |
| `ANY_ATTACK` | -0.0404 | [-0.0618, -0.0237] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9981 | [-0.9993, -0.9967] | **yes** |
| `FRG` | -0.4513 | [-0.4742, -0.4313] | **yes** |
| `SAG.PB` | -0.9201 | [-0.9272, -0.9120] | **yes** |
| `SAG.PBM` | -0.5010 | [-0.5380, -0.4588] | **yes** |
| `ANY_ATTACK` | -0.1824 | [-0.2418, -0.1289] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.2689 | [-0.2751, -0.2633] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9889 | [-0.9966, -0.9787] | **yes** |
| `ANY_ATTACK` | -0.5359 | [-0.5893, -0.4848] | **yes** |

### `f10-xgboost-no-abs-no-counters` (A) vs `d2-rule-stnum-gap` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8398 | [-0.8826, -0.7784] | **yes** |
| `FRG` | -0.5848 | [-0.6819, -0.4870] | **yes** |
| `SAG.PB` | -0.8242 | [-0.8538, -0.7889] | **yes** |
| `SAG.PBM` | -0.3667 | [-0.4169, -0.3173] | **yes** |
| `ANY_ATTACK` | -0.6052 | [-0.6423, -0.5622] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0214 | [-0.0370, -0.0087] | **yes** |
| `FRG` | -0.0107 | [-0.0152, -0.0072] | **yes** |
| `SAG.PB` | -0.0243 | [-0.0351, -0.0151] | **yes** |
| `SAG.PBM` | -0.0211 | [-0.0307, -0.0124] | **yes** |
| `ANY_ATTACK` | -0.0002 | [-0.0037, +0.0032] | no |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2095 | [-0.2744, -0.1442] | **yes** |
| `FRG` | -0.1100 | [-0.1345, -0.0875] | **yes** |
| `SAG.PB` | -0.2418 | [-0.3083, -0.1829] | **yes** |
| `SAG.PBM` | -0.1608 | [-0.1754, -0.1468] | **yes** |
| `ANY_ATTACK` | +0.0722 | [+0.0309, +0.1137] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9981 | [-0.9993, -0.9967] | **yes** |
| `FRG` | -0.7991 | [-0.8096, -0.7875] | **yes** |
| `SAG.PB` | -0.9201 | [-0.9272, -0.9120] | **yes** |
| `SAG.PBM` | -0.5010 | [-0.5380, -0.4588] | **yes** |
| `ANY_ATTACK` | -0.2924 | [-0.3369, -0.2485] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.9726 | [-0.9743, -0.9710] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9889 | [-0.9966, -0.9787] | **yes** |
| `ANY_ATTACK` | -0.5744 | [-0.6323, -0.5153] | **yes** |

### `f10-xgboost-no-abs-no-counters` (A) vs `d2-rule-interval-t` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8398 | [-0.8826, -0.7784] | **yes** |
| `FRG` | -0.5848 | [-0.6819, -0.4870] | **yes** |
| `SAG.PB` | -0.8242 | [-0.8538, -0.7889] | **yes** |
| `SAG.PBM` | -0.3667 | [-0.4169, -0.3173] | **yes** |
| `ANY_ATTACK` | -0.7234 | [-0.7507, -0.6948] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0214 | [-0.0370, -0.0087] | **yes** |
| `FRG` | -0.0106 | [-0.0152, -0.0070] | **yes** |
| `SAG.PB` | -0.0243 | [-0.0351, -0.0151] | **yes** |
| `SAG.PBM` | -0.0211 | [-0.0307, -0.0124] | **yes** |
| `ANY_ATTACK` | -0.0045 | [-0.0070, -0.0024] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2095 | [-0.2744, -0.1442] | **yes** |
| `FRG` | -0.1152 | [-0.1426, -0.0903] | **yes** |
| `SAG.PB` | -0.2418 | [-0.3083, -0.1829] | **yes** |
| `SAG.PBM` | -0.1608 | [-0.1754, -0.1468] | **yes** |
| `ANY_ATTACK` | -0.0420 | [-0.0625, -0.0249] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9981 | [-0.9993, -0.9967] | **yes** |
| `FRG` | -0.7945 | [-0.8030, -0.7844] | **yes** |
| `SAG.PB` | -0.9201 | [-0.9272, -0.9120] | **yes** |
| `SAG.PBM` | -0.5010 | [-0.5380, -0.4588] | **yes** |
| `ANY_ATTACK` | -0.3918 | [-0.4343, -0.3524] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.8962 | [-0.8989, -0.8935] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9889 | [-0.9966, -0.9787] | **yes** |
| `ANY_ATTACK` | -0.4588 | [-0.5139, -0.4019] | **yes** |

### `f10-xgboost-no-abs-no-counters` (A) vs `d2-rule-sqnum-gap` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8398 | [-0.8826, -0.7784] | **yes** |
| `FRG` | -0.5390 | [-0.6318, -0.4488] | **yes** |
| `SAG.PB` | -0.8242 | [-0.8538, -0.7889] | **yes** |
| `SAG.PBM` | -0.3667 | [-0.4169, -0.3173] | **yes** |
| `ANY_ATTACK` | -0.7793 | [-0.8094, -0.7472] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0214 | [-0.0370, -0.0087] | **yes** |
| `FRG` | -0.0107 | [-0.0152, -0.0071] | **yes** |
| `SAG.PB` | -0.0243 | [-0.0351, -0.0151] | **yes** |
| `SAG.PBM` | -0.0211 | [-0.0307, -0.0124] | **yes** |
| `ANY_ATTACK` | -0.0052 | [-0.0078, -0.0032] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2095 | [-0.2744, -0.1442] | **yes** |
| `FRG` | -0.1158 | [-0.1432, -0.0910] | **yes** |
| `SAG.PB` | -0.2418 | [-0.3083, -0.1829] | **yes** |
| `SAG.PBM` | -0.1608 | [-0.1754, -0.1468] | **yes** |
| `ANY_ATTACK` | -0.0491 | [-0.0699, -0.0318] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9981 | [-0.9993, -0.9967] | **yes** |
| `FRG` | -0.8028 | [-0.8121, -0.7923] | **yes** |
| `SAG.PB` | -0.9201 | [-0.9272, -0.9120] | **yes** |
| `SAG.PBM` | -0.5010 | [-0.5380, -0.4588] | **yes** |
| `ANY_ATTACK` | -0.4774 | [-0.5308, -0.4272] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.2240 | [-0.2283, -0.2200] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9889 | [-0.9966, -0.9787] | **yes** |
| `ANY_ATTACK` | -0.6261 | [-0.6755, -0.5778] | **yes** |

### `f10-xgboost-no-abs-no-counters` (A) vs `d2-rule-delay` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8398 | [-0.8826, -0.7784] | **yes** |
| `FRG` | -0.5862 | [-0.6833, -0.4882] | **yes** |
| `SAG.PB` | -0.8242 | [-0.8538, -0.7889] | **yes** |
| `SAG.PBM` | -0.3667 | [-0.4169, -0.3173] | **yes** |
| `ANY_ATTACK` | -0.8068 | [-0.8365, -0.7770] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0214 | [-0.0370, -0.0087] | **yes** |
| `FRG` | -0.0108 | [-0.0153, -0.0072] | **yes** |
| `SAG.PB` | -0.0243 | [-0.0351, -0.0151] | **yes** |
| `SAG.PBM` | -0.0211 | [-0.0307, -0.0124] | **yes** |
| `ANY_ATTACK` | -0.0052 | [-0.0078, -0.0032] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2095 | [-0.2744, -0.1442] | **yes** |
| `FRG` | -0.1168 | [-0.1443, -0.0919] | **yes** |
| `SAG.PB` | -0.2418 | [-0.3083, -0.1829] | **yes** |
| `SAG.PBM` | -0.1608 | [-0.1754, -0.1468] | **yes** |
| `ANY_ATTACK` | -0.0493 | [-0.0701, -0.0320] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9981 | [-0.9993, -0.9967] | **yes** |
| `FRG` | -0.8128 | [-0.8215, -0.8027] | **yes** |
| `SAG.PB` | -0.9201 | [-0.9272, -0.9120] | **yes** |
| `SAG.PBM` | -0.5010 | [-0.5380, -0.4588] | **yes** |
| `ANY_ATTACK` | -0.4793 | [-0.5318, -0.4297] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.9999 | [-1.0000, -0.9998] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9889 | [-0.9966, -0.9787] | **yes** |
| `ANY_ATTACK` | -0.9691 | [-0.9787, -0.9576] | **yes** |

### `f10-xgboost-no-abs-no-counters` (A) vs `d2-rule-time-since-change` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.8398 | [-0.8826, -0.7784] | **yes** |
| `FRG` | -0.5859 | [-0.6830, -0.4879] | **yes** |
| `SAG.PB` | -0.8242 | [-0.8538, -0.7889] | **yes** |
| `SAG.PBM` | -0.3667 | [-0.4169, -0.3173] | **yes** |
| `ANY_ATTACK` | -0.8116 | [-0.8414, -0.7819] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0214 | [-0.0370, -0.0087] | **yes** |
| `FRG` | -0.0107 | [-0.0152, -0.0071] | **yes** |
| `SAG.PB` | -0.0243 | [-0.0351, -0.0151] | **yes** |
| `SAG.PBM` | -0.0211 | [-0.0307, -0.0124] | **yes** |
| `ANY_ATTACK` | -0.0052 | [-0.0078, -0.0032] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.2095 | [-0.2744, -0.1442] | **yes** |
| `FRG` | -0.1164 | [-0.1438, -0.0915] | **yes** |
| `SAG.PB` | -0.2418 | [-0.3083, -0.1829] | **yes** |
| `SAG.PBM` | -0.1608 | [-0.1754, -0.1468] | **yes** |
| `ANY_ATTACK` | -0.0493 | [-0.0701, -0.0320] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.9981 | [-0.9993, -0.9967] | **yes** |
| `FRG` | -0.8040 | [-0.8132, -0.7934] | **yes** |
| `SAG.PB` | -0.9201 | [-0.9272, -0.9120] | **yes** |
| `SAG.PBM` | -0.5010 | [-0.5380, -0.4588] | **yes** |
| `ANY_ATTACK` | -0.4778 | [-0.5310, -0.4276] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `FRG` | -0.9110 | [-0.9163, -0.9054] | **yes** |
| `SAG.PB` | -1.0000 | [-1.0000, -1.0000] | **yes** |
| `SAG.PBM` | -0.9889 | [-0.9966, -0.9787] | **yes** |
| `ANY_ATTACK` | -0.9563 | [-0.9650, -0.9471] | **yes** |

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

