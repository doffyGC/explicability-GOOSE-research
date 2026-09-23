# Threshold curves over the grouped folds

- Generated: 2026-09-22 21:03:55 UTC
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

## Paired comparison

Each replicate draws one set of runs and scores **both** configurations
on it, so the run-to-run variation they share cancels instead of being
counted twice. Two marginal intervals overlapping does not mean two
configurations are indistinguishable; this is the table that settles it.
Each configuration keeps its own cross-fold thresholds - the question is
which is better when each is operated properly, not which wins at a
threshold borrowed from the other.

### `v2-xgboost-none` (A) vs `f10-xgboost-no-abs-no-counters` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0395 | [-0.0651, -0.0221] | **yes** |
| `FRG` | -0.0481 | [-0.0760, -0.0175] | **yes** |
| `SAG.PB` | -0.0266 | [-0.0364, -0.0178] | **yes** |
| `SAG.PBM` | -0.0378 | [-0.0448, -0.0311] | **yes** |
| `ANY_ATTACK` | -0.0085 | [-0.0119, -0.0058] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0017 | [-0.0222, +0.0198] | no |
| `FRG` | -0.0089 | [-0.0292, +0.0064] | no |
| `SAG.PB` | -0.0002 | [-0.0166, +0.0134] | no |
| `SAG.PBM` | -0.0008 | [-0.0141, +0.0123] | no |
| `ANY_ATTACK` | -0.0008 | [-0.0028, +0.0009] | no |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0081 | [-0.0528, +0.0340] | no |
| `FRG` | -0.0169 | [-0.0543, +0.0205] | no |
| `SAG.PB` | +0.0062 | [-0.0382, +0.0519] | no |
| `SAG.PBM` | -0.0094 | [-0.0211, +0.0007] | no |
| `ANY_ATTACK` | -0.0016 | [-0.0091, +0.0055] | no |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0009 | [-0.0028, +0.0009] | no |
| `FRG` | -0.0058 | [-0.0128, +0.0022] | no |
| `SAG.PB` | -0.0017 | [-0.0037, +0.0002] | no |
| `SAG.PBM` | -0.0095 | [-0.0135, -0.0059] | **yes** |
| `ANY_ATTACK` | -0.0029 | [-0.0160, +0.0089] | no |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0001 | [-0.0001, +0.0005] | no |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | -0.0092 | [-0.0172, -0.0030] | **yes** |
| `ANY_ATTACK` | -0.0016 | [-0.0026, -0.0007] | **yes** |

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

