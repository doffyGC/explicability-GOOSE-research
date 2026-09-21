# Threshold curves over the grouped folds

- Generated: 2026-09-21 16:48:56 UTC
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

## `d1-xgboost-no-delta`

- model: `xgboost` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `3109e4d480524d8a`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,779 | 0.4592% | 45 | 0.1126 [0.0772, 0.1545] | 24.5x |
| `FRG` | 63,963 | 0.5785% | 45 | 0.0320 [0.0215, 0.0456] | 5.5x |
| `SAG.PB` | 46,959 | 0.4247% | 45 | 0.0256 [0.0200, 0.0324] | 6.0x |
| `SAG.PBM` | 54,828 | 0.4958% | 45 | 0.0342 [0.0250, 0.0454] | 6.9x |
| `ANY_ATTACK` | 216,529 | 1.9582% | 180 | 0.1101 [0.0906, 0.1344] | 5.6x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.54627656, 0.712723176, 0.739896819, ... | 0.0120% | 0.0062 [0.0000, 0.0156] | 0.2352 [0.0025, 0.5000] | 1,331 | 313 |
| `FRG` | 0.145736236, 0.169021991, 0.177416178, ... | 0.0132% | 0.0010 [0.0004, 0.0017] | 0.0425 [0.0136, 0.1457] | 1,460 | 62 |
| `SAG.PB` | 0.0592438503, 0.0625967842, 0.0637532549, ... | 0.0087% | 0.0001 [0.0000, 0.0002] | 0.0031 [0.0000, 0.0206] | 961 | 3 |
| `SAG.PBM` | 0.0649296114, 0.0673431316, 0.0711197951 | 0.0108% | 0.0005 [0.0001, 0.0011] | 0.0226 [0.0059, 0.0641] | 1,196 | 27 |
| `ANY_ATTACK` | 0.584667486, 0.77229215, 0.78574453, ... | 0.0123% | 0.0019 [0.0000, 0.0047] | 0.2987 [0.0407, 0.5000] | 1,359 | 406 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 13 (1.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1,005 (98.7%) |
| `FRG` | 119 (8.5%) | — | 0 (0.0%) | 0 (0.0%) | 54 (3.9%) | 1,225 (87.6%) |
| `SAG.PB` | 266 (27.8%) | 0 (0.0%) | — | 1 (0.1%) | 0 (0.0%) | 691 (72.1%) |
| `SAG.PBM` | 163 (13.9%) | 30 (2.6%) | 0 (0.0%) | — | 44 (3.8%) | 932 (79.7%) |
| `ANY_ATTACK` | — | — | — | — | 0 (0.0%) | 953 (100.0%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.192127014, 0.198266117, 0.23465365, ... | 0.1151% | 0.0530 [0.0243, 0.0869] | 0.2116 [0.1055, 0.3310] | 12,725 | 2,692 |
| `FRG` | 0.0737452658, 0.0914512095, 0.0930878262, ... | 0.1183% | 0.0065 [0.0029, 0.0109] | 0.0319 [0.0112, 0.1000] | 13,081 | 417 |
| `SAG.PB` | 0.0501688886, 0.0520642831 | 0.0875% | 0.0026 [0.0008, 0.0055] | 0.0124 [0.0036, 0.0281] | 9,680 | 120 |
| `SAG.PBM` | 0.0550346565, 0.0571028293, 0.0581640975 | 0.1381% | 0.0062 [0.0028, 0.0102] | 0.0221 [0.0098, 0.0380] | 15,267 | 338 |
| `ANY_ATTACK` | 0.245344963, 0.248980934, 0.283292557, ... | 0.1124% | 0.0146 [0.0068, 0.0235] | 0.2538 [0.1507, 0.3784] | 12,424 | 3,153 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 240 (2.4%) | 59 (0.6%) | 32 (0.3%) | 21 (0.2%) | 9,681 (96.5%) |
| `FRG` | 555 (4.4%) | — | 0 (0.0%) | 0 (0.0%) | 305 (2.4%) | 11,804 (93.2%) |
| `SAG.PB` | 292 (3.1%) | 0 (0.0%) | — | 269 (2.8%) | 1 (0.0%) | 8,998 (94.1%) |
| `SAG.PBM` | 434 (2.9%) | 109 (0.7%) | 72 (0.5%) | — | 219 (1.5%) | 14,095 (94.4%) |
| `ANY_ATTACK` | — | — | — | — | 57 (0.6%) | 9,214 (99.4%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.098156326, 0.0998997437, 0.101670636 | 0.9643% | 0.2558 [0.1964, 0.3229] | 0.1218 [0.0822, 0.1637] | 106,625 | 12,989 |
| `FRG` | 0.0393573633, 0.0401028552, 0.041634622 | 1.0229% | 0.0864 [0.0558, 0.1227] | 0.0488 [0.0293, 0.0719] | 113,107 | 5,524 |
| `SAG.PB` | 0.0401028552, 0.041634622 | 1.0344% | 0.0770 [0.0553, 0.0990] | 0.0316 [0.0213, 0.0419] | 114,383 | 3,615 |
| `SAG.PBM` | 0.0492458741, 0.0501688886, 0.0511082732 | 1.0236% | 0.0668 [0.0416, 0.0950] | 0.0324 [0.0194, 0.0474] | 113,179 | 3,665 |
| `ANY_ATTACK` | 0.163603621, 0.16629514, 0.169021991 | 1.0000% | 0.0903 [0.0646, 0.1208] | 0.1768 [0.1348, 0.2241] | 110,573 | 19,550 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 1,246 (1.3%) | 1,408 (1.5%) | 2,496 (2.7%) | 1,093 (1.2%) | 87,393 (93.3%) |
| `FRG` | 2,322 (2.2%) | — | 232 (0.2%) | 872 (0.8%) | 5,391 (5.0%) | 98,766 (91.8%) |
| `SAG.PB` | 434 (0.4%) | 0 (0.0%) | — | 1,873 (1.7%) | 2 (0.0%) | 108,459 (97.9%) |
| `SAG.PBM` | 2,962 (2.7%) | 518 (0.5%) | 1,015 (0.9%) | — | 1,835 (1.7%) | 103,184 (94.2%) |
| `ANY_ATTACK` | — | — | — | — | 1,570 (1.7%) | 89,453 (98.3%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.000979771865, 0.0013130369, 0.00182945648, ... | 9.9764% | 0.9957 [0.9912, 0.9986] | 0.0458 [0.0336, 0.0596] | 1,103,138 | 50,559 |
| `FRG` | 0.0227183406, 0.0231562598, 0.0236024165 | 9.7342% | 0.5191 [0.4309, 0.5959] | 0.0308 [0.0216, 0.0410] | 1,076,352 | 33,205 |
| `SAG.PB` | 0.0167205476, 0.0173753713, 0.0177121659 | 9.9036% | 0.5547 [0.4961, 0.6128] | 0.0238 [0.0186, 0.0297] | 1,095,086 | 26,050 |
| `SAG.PBM` | 0.0170448696, 0.0180553688, 0.0191246095 | 9.8723% | 0.7085 [0.6372, 0.7760] | 0.0356 [0.0266, 0.0459] | 1,091,629 | 38,844 |
| `ANY_ATTACK` | 0.0560597725, 0.0571028293, 0.0592438503 | 9.9380% | 0.5084 [0.4669, 0.5521] | 0.1002 [0.0867, 0.1149] | 1,098,896 | 110,094 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 13,797 (1.3%) | 12,437 (1.2%) | 27,396 (2.6%) | 31,454 (3.0%) | 967,495 (91.9%) |
| `FRG` | 12,650 (1.2%) | — | 3,363 (0.3%) | 11,951 (1.1%) | 78,971 (7.6%) | 936,212 (89.7%) |
| `SAG.PB` | 16,136 (1.5%) | 908 (0.1%) | — | 12,519 (1.2%) | 2,713 (0.3%) | 1,036,760 (97.0%) |
| `SAG.PBM` | 32,172 (3.1%) | 14,287 (1.4%) | 12,022 (1.1%) | — | 17,535 (1.7%) | 976,769 (92.8%) |
| `ANY_ATTACK` | — | — | — | — | 24,862 (2.5%) | 963,940 (97.5%) |

## Paired comparison

Each replicate draws one set of runs and scores **both** configurations
on it, so the run-to-run variation they share cancels instead of being
counted twice. Two marginal intervals overlapping does not mean two
configurations are indistinguishable; this is the table that settles it.
Each configuration keeps its own cross-fold thresholds - the question is
which is better when each is operated properly, not which wins at a
threshold borrowed from the other.

### `d2-rule-interval-timestamp` (A) vs `d1-xgboost-no-delta` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.1080 | [+0.0736, +0.1490] | **yes** |
| `FRG` | -0.1138 | [-0.1489, -0.0793] | **yes** |
| `SAG.PB` | +0.0214 | [+0.0163, +0.0273] | **yes** |
| `SAG.PBM` | +0.0292 | [+0.0209, +0.0389] | **yes** |
| `ANY_ATTACK` | -0.1310 | [-0.1948, -0.0724] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0062 | [+0.0000, +0.0156] | **yes** |
| `FRG` | +0.0010 | [+0.0004, +0.0017] | **yes** |
| `SAG.PB` | +0.0001 | [+0.0000, +0.0002] | no |
| `SAG.PBM` | +0.0005 | [+0.0001, +0.0011] | **yes** |
| `ANY_ATTACK` | +0.0014 | [-0.0005, +0.0042] | no |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0530 | [+0.0243, +0.0869] | **yes** |
| `FRG` | +0.0063 | [+0.0027, +0.0107] | **yes** |
| `SAG.PB` | +0.0026 | [+0.0008, +0.0055] | **yes** |
| `SAG.PBM` | +0.0062 | [+0.0028, +0.0102] | **yes** |
| `ANY_ATTACK` | +0.0056 | [-0.0025, +0.0156] | no |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.2558 | [+0.1964, +0.3229] | **yes** |
| `FRG` | -0.2751 | [-0.3155, -0.2299] | **yes** |
| `SAG.PB` | +0.0770 | [+0.0553, +0.0990] | **yes** |
| `SAG.PBM` | +0.0668 | [+0.0416, +0.0950] | **yes** |
| `ANY_ATTACK` | -0.2077 | [-0.2639, -0.1459] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.9957 | [+0.9912, +0.9986] | **yes** |
| `FRG` | -0.2119 | [-0.2959, -0.1401] | **yes** |
| `SAG.PB` | +0.5547 | [+0.4961, +0.6128] | **yes** |
| `SAG.PBM` | +0.7085 | [+0.6372, +0.7760] | **yes** |
| `ANY_ATTACK` | +0.0618 | [-0.0176, +0.1476] | no |

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

