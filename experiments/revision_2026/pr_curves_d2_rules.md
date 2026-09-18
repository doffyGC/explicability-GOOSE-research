# Threshold curves over the grouped folds

- Generated: 2026-09-18 17:47:59 UTC
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

## Paired comparison

Each replicate draws one set of runs and scores **both** configurations
on it, so the run-to-run variation they share cancels instead of being
counted twice. Two marginal intervals overlapping does not mean two
configurations are indistinguishable; this is the table that settles it.
Each configuration keeps its own cross-fold thresholds - the question is
which is better when each is operated properly, not which wins at a
threshold borrowed from the other.

### `d2-rule-interval-timestamp` (A) vs `d2-rule-stnum-gap` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no - indistinguishable |
| `FRG` | -0.1391 | [-0.1792, -0.1014] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no - indistinguishable |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no - indistinguishable |
| `ANY_ATTACK` | -0.0219 | [-0.0884, +0.0452] | no - indistinguishable |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0000 | [+0.0000, +0.0001] | no |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | +0.0046 | [+0.0025, +0.0071] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0066 | [+0.0031, +0.0105] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | +0.1126 | [+0.0779, +0.1512] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | -0.3478 | [-0.3675, -0.3263] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | -0.1100 | [-0.1642, -0.0515] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | -0.7038 | [-0.7096, -0.6969] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | -0.0385 | [-0.1198, +0.0522] | no |

### `d2-rule-interval-timestamp` (A) vs `d2-rule-sqnum-gap` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no - indistinguishable |
| `FRG` | -0.0934 | [-0.1240, -0.0649] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no - indistinguishable |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no - indistinguishable |
| `ANY_ATTACK` | -0.1960 | [-0.2535, -0.1481] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0001 | [+0.0000, +0.0002] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | -0.0004 | [-0.0007, -0.0002] | **yes** |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0008 | [+0.0005, +0.0010] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | -0.0087 | [-0.0124, -0.0054] | **yes** |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | -0.3515 | [-0.3744, -0.3258] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | -0.2950 | [-0.3427, -0.2498] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0000 | [+0.0000, +0.0000] | no |
| `FRG` | +0.0449 | [+0.0419, +0.0484] | **yes** |
| `SAG.PB` | +0.0000 | [+0.0000, +0.0000] | no |
| `SAG.PBM` | +0.0000 | [+0.0000, +0.0000] | no |
| `ANY_ATTACK` | -0.0902 | [-0.1384, -0.0474] | **yes** |

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

