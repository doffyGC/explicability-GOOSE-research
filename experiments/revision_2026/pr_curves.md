# Threshold curves over the grouped folds

- Generated: 2026-09-13 14:19:34 UTC
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

## `d5-xgboost-none`

- model: `xgboost` | balance: `none` | train cap: — | status: `full_grouped_run`
- posterior prior: **as trained (uncorrected)** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `e0c172eac5276c33`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,782 | 0.2186% | 45 | 0.0620 [0.0446, 0.0823] | 28.4x |
| `FRG` | 63,976 | 0.2754% | 45 | 0.0255 [0.0172, 0.0352] | 9.3x |
| `SAG.PB` | 46,959 | 0.2022% | 45 | 0.0231 [0.0154, 0.0333] | 11.4x |
| `SAG.PBM` | 54,828 | 0.2361% | 45 | 0.0298 [0.0213, 0.0393] | 12.6x |
| `ANY_ATTACK` | 216,545 | 0.9323% | 180 | 0.0724 [0.0617, 0.0844] | 7.8x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.241744999, 0.248980934, 0.283292557, ... | 0.0109% | 0.0024 [0.0004, 0.0051] | 0.0491 [0.0131, 0.1616] | 2,524 | 124 |
| `FRG` | 0.0998997437, 0.103469313, 0.10529608 | 0.0101% | 0.0034 [0.0026, 0.0043] | 0.0926 [0.0603, 0.1311] | 2,353 | 218 |
| `SAG.PB` | 0.0998997437, 0.103469313, 0.107151244 | 0.0104% | 0.0075 [0.0031, 0.0123] | 0.1460 [0.0612, 0.2325] | 2,424 | 354 |
| `SAG.PBM` | 0.101670636, 0.103469313, 0.10529608, ... | 0.0106% | 0.0054 [0.0033, 0.0077] | 0.1202 [0.0721, 0.1660] | 2,454 | 295 |
| `ANY_ATTACK` | 0.279341892, 0.29129435, 0.311867701, ... | 0.0107% | 0.0010 [0.0005, 0.0017] | 0.0887 [0.0541, 0.2354] | 2,491 | 221 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 23 (1.0%) | 1 (0.0%) | 0 (0.0%) | 1 (0.0%) | 2,375 (99.0%) |
| `FRG` | 19 (0.9%) | — | 40 (1.9%) | 134 (6.3%) | 180 (8.4%) | 1,762 (82.5%) |
| `SAG.PB` | 0 (0.0%) | 4 (0.2%) | — | 174 (8.4%) | 34 (1.6%) | 1,858 (89.8%) |
| `SAG.PBM` | 0 (0.0%) | 110 (5.1%) | 124 (5.7%) | — | 255 (11.8%) | 1,670 (77.4%) |
| `ANY_ATTACK` | — | — | — | — | 36 (1.6%) | 2,234 (98.4%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.098156326, 0.101670636, 0.10529608, ... | 0.0993% | 0.0514 [0.0315, 0.0743] | 0.1131 [0.0695, 0.1691] | 23,073 | 2,609 |
| `FRG` | 0.0530371761, 0.0560597725, 0.0592438503, ... | 0.1017% | 0.0221 [0.0169, 0.0283] | 0.0599 [0.0365, 0.0942] | 23,631 | 1,415 |
| `SAG.PB` | 0.0540272129, 0.0550346565, 0.0560597725, ... | 0.0989% | 0.0362 [0.0213, 0.0517] | 0.0739 [0.0417, 0.1096] | 22,978 | 1,699 |
| `SAG.PBM` | 0.0592438503, 0.0603423634, 0.0614599146, ... | 0.0981% | 0.0368 [0.0303, 0.0435] | 0.0885 [0.0645, 0.1138] | 22,780 | 2,016 |
| `ANY_ATTACK` | 0.153187162, 0.158325854, 0.160947254, ... | 0.0991% | 0.0179 [0.0141, 0.0222] | 0.1682 [0.1327, 0.2139] | 23,024 | 3,872 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 221 (1.1%) | 212 (1.0%) | 168 (0.8%) | 68 (0.3%) | 19,795 (96.7%) |
| `FRG` | 295 (1.3%) | — | 257 (1.2%) | 733 (3.3%) | 982 (4.4%) | 19,949 (89.8%) |
| `SAG.PB` | 0 (0.0%) | 78 (0.4%) | — | 985 (4.6%) | 232 (1.1%) | 19,984 (93.9%) |
| `SAG.PBM` | 25 (0.1%) | 845 (4.1%) | 747 (3.6%) | — | 1,508 (7.3%) | 17,639 (84.9%) |
| `ANY_ATTACK` | — | — | — | — | 829 (4.3%) | 18,323 (95.7%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.0511082732, 0.0540272129, 0.0550346565 | 0.9888% | 0.2960 [0.2417, 0.3508] | 0.0655 [0.0461, 0.0863] | 229,656 | 15,033 |
| `FRG` | 0.0274863543, 0.0280135474, 0.0285505552 | 0.9887% | 0.1356 [0.1053, 0.1671] | 0.0378 [0.0257, 0.0513] | 229,635 | 8,676 |
| `SAG.PB` | 0.0214525718, 0.021866638, 0.0227183406 | 0.9826% | 0.1445 [0.1021, 0.1876] | 0.0297 [0.0202, 0.0397] | 228,218 | 6,787 |
| `SAG.PBM` | 0.0259619894, 0.0264607404, 0.0269688074, ... | 0.9827% | 0.1764 [0.1532, 0.2005] | 0.0424 [0.0310, 0.0547] | 228,246 | 9,669 |
| `ANY_ATTACK` | 0.0898405165, 0.0930878262, 0.0947506774 | 0.9701% | 0.1337 [0.1161, 0.1529] | 0.1285 [0.1105, 0.1471] | 225,326 | 28,959 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 2,200 (1.0%) | 2,992 (1.4%) | 4,200 (2.0%) | 2,349 (1.1%) | 202,882 (94.5%) |
| `FRG` | 3,163 (1.4%) | — | 801 (0.4%) | 2,648 (1.2%) | 4,556 (2.1%) | 209,791 (94.9%) |
| `SAG.PB` | 262 (0.1%) | 735 (0.3%) | — | 3,778 (1.7%) | 1,247 (0.6%) | 215,409 (97.3%) |
| `SAG.PBM` | 3,557 (1.6%) | 3,391 (1.6%) | 3,202 (1.5%) | — | 5,170 (2.4%) | 203,257 (93.0%) |
| `ANY_ATTACK` | — | — | — | — | 4,277 (2.2%) | 192,090 (97.8%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.000731031709, 0.000854622182, 0.000871471712, ... | 9.9976% | 0.9947 [0.9899, 0.9981] | 0.0218 [0.0161, 0.0277] | 2,322,105 | 50,512 |
| `FRG` | 0.00867461467, 0.00884427658, 0.00937324245, ... | 10.0210% | 0.6744 [0.5916, 0.7496] | 0.0185 [0.0131, 0.0242] | 2,327,537 | 43,143 |
| `SAG.PB` | 0.00772233139, 0.00787351607, 0.00802763663, ... | 9.9598% | 0.6633 [0.6210, 0.7055] | 0.0135 [0.0110, 0.0162] | 2,313,322 | 31,148 |
| `SAG.PBM` | 0.00802763663, 0.00834491071, 0.00884427658 | 9.9059% | 0.6984 [0.6370, 0.7549] | 0.0166 [0.0125, 0.0214] | 2,300,803 | 38,294 |
| `ANY_ATTACK` | 0.0240569577, 0.0249917932, 0.0254723933 | 9.8023% | 0.5625 [0.5262, 0.6006] | 0.0535 [0.0473, 0.0595] | 2,276,735 | 121,799 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 13,888 (0.6%) | 12,306 (0.5%) | 27,044 (1.2%) | 29,745 (1.3%) | 2,188,610 (96.3%) |
| `FRG` | 13,870 (0.6%) | — | 4,004 (0.2%) | 11,814 (0.5%) | 48,100 (2.1%) | 2,206,606 (96.6%) |
| `SAG.PB` | 26,690 (1.2%) | 4,415 (0.2%) | — | 15,450 (0.7%) | 7,046 (0.3%) | 2,228,573 (97.7%) |
| `SAG.PBM` | 32,202 (1.4%) | 13,755 (0.6%) | 14,486 (0.6%) | — | 17,761 (0.8%) | 2,184,305 (96.5%) |
| `ANY_ATTACK` | — | — | — | — | 27,298 (1.3%) | 2,127,638 (98.7%) |

## `d5-xgboost-downsample`

- model: `xgboost` | balance: `downsample` | train cap: — | status: `full_grouped_run`
- posterior prior: **corrected to the pool's natural prior** | threshold selection: **cross-fold (calibrated on the other folds)**
- runs: 265 | dataset SHA-256: `e0c172eac5276c33`

### Threshold-free ranking

| Target | positive rows | prevalence | runs carrying it | AP [95% CI] | AP / prevalence |
|---|---:|---:|---:|---|---:|
| `SAG.DB` | 50,782 | 0.2186% | 45 | 0.0593 [0.0430, 0.0759] | 27.1x |
| `FRG` | 63,976 | 0.2754% | 45 | 0.0209 [0.0140, 0.0294] | 7.6x |
| `SAG.PB` | 46,959 | 0.2022% | 45 | 0.0187 [0.0138, 0.0248] | 9.2x |
| `SAG.PBM` | 54,828 | 0.2361% | 45 | 0.0247 [0.0178, 0.0325] | 10.5x |
| `ANY_ATTACK` | 216,545 | 0.9323% | 180 | 0.0625 [0.0535, 0.0730] | 6.7x |

### At 1 alert per 10,000 messages (budget 0.0001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.207749564, 0.248980934, 0.26769462, ... | 0.0134% | 0.0069 [0.0005, 0.0166] | 0.1121 [0.0229, 0.1732] | 3,123 | 350 |
| `FRG` | 0.082165036, 0.125175259, 0.12733082, ... | 0.0262% | 0.0006 [0.0002, 0.0010] | 0.0061 [0.0018, 0.0618] | 6,083 | 37 |
| `SAG.PB` | 0.0673431316, 0.0711197951, 0.0724215626 | 0.0109% | 0.0012 [0.0002, 0.0024] | 0.0221 [0.0050, 0.0457] | 2,538 | 56 |
| `SAG.PBM` | 0.0571028293, 0.0625967842, 0.0637532549 | 0.0154% | 0.0034 [0.0016, 0.0055] | 0.0528 [0.0251, 0.0843] | 3,578 | 189 |
| `ANY_ATTACK` | 0.220908304, 0.263881388, 0.279341892, ... | 0.0139% | 0.0017 [0.0002, 0.0041] | 0.1130 [0.0362, 0.1582] | 3,229 | 365 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 9 (0.3%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 2,764 (99.7%) |
| `FRG` | 84 (1.4%) | — | 0 (0.0%) | 0 (0.0%) | 5 (0.1%) | 5,957 (98.5%) |
| `SAG.PB` | 0 (0.0%) | 0 (0.0%) | — | 41 (1.7%) | 1 (0.0%) | 2,440 (98.3%) |
| `SAG.PBM` | 3 (0.1%) | 66 (1.9%) | 18 (0.5%) | — | 148 (4.4%) | 3,154 (93.1%) |
| `ANY_ATTACK` | — | — | — | — | 1 (0.0%) | 2,863 (100.0%) |

### At 10 alerts per 10,000 messages (budget 0.001)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.0851608719, 0.098156326, 0.0998997437, ... | 0.1078% | 0.0544 [0.0287, 0.0852] | 0.1103 [0.0683, 0.1497] | 25,031 | 2,761 |
| `FRG` | 0.0474479477, 0.0571028293, 0.0592438503, ... | 0.1195% | 0.0098 [0.0063, 0.0139] | 0.0226 [0.0099, 0.0602] | 27,758 | 627 |
| `SAG.PB` | 0.0483389771, 0.0501688886, 0.0511082732, ... | 0.0971% | 0.0218 [0.0126, 0.0318] | 0.0454 [0.0249, 0.0690] | 22,564 | 1,024 |
| `SAG.PBM` | 0.0432222617, 0.0440376051, 0.0448676079, ... | 0.0899% | 0.0233 [0.0178, 0.0285] | 0.0611 [0.0425, 0.0811] | 20,884 | 1,276 |
| `ANY_ATTACK` | 0.11289016, 0.12733082, 0.13173709 | 0.1118% | 0.0148 [0.0084, 0.0235] | 0.1230 [0.0946, 0.1538] | 25,970 | 3,195 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 110 (0.5%) | 174 (0.8%) | 148 (0.7%) | 34 (0.2%) | 21,804 (97.9%) |
| `FRG` | 310 (1.1%) | — | 46 (0.2%) | 91 (0.3%) | 193 (0.7%) | 26,491 (97.6%) |
| `SAG.PB` | 0 (0.0%) | 3 (0.0%) | — | 411 (1.9%) | 7 (0.0%) | 21,119 (98.0%) |
| `SAG.PBM` | 53 (0.3%) | 438 (2.2%) | 323 (1.6%) | — | 754 (3.8%) | 18,040 (92.0%) |
| `ANY_ATTACK` | — | — | — | — | 114 (0.5%) | 22,661 (99.5%) |

### At 100 alerts per 10,000 messages (budget 0.01)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.0440376051, 0.0457125062, 0.0474479477, ... | 0.9821% | 0.2839 [0.2237, 0.3436] | 0.0632 [0.0444, 0.0823] | 228,115 | 14,415 |
| `FRG` | 0.0249917932, 0.0264607404, 0.0274863543, ... | 1.0372% | 0.1142 [0.0877, 0.1428] | 0.0303 [0.0198, 0.0426] | 240,904 | 7,303 |
| `SAG.PB` | 0.0236024165, 0.0240569577, 0.0245200327 | 0.9478% | 0.1326 [0.0977, 0.1689] | 0.0283 [0.0200, 0.0370] | 220,148 | 6,225 |
| `SAG.PBM` | 0.0240569577, 0.0245200327, 0.0249917932, ... | 0.9594% | 0.1510 [0.1302, 0.1708] | 0.0372 [0.0276, 0.0477] | 222,832 | 8,279 |
| `ANY_ATTACK` | 0.0750912047, 0.0764596808, 0.0778509971, ... | 0.9832% | 0.1123 [0.0937, 0.1324] | 0.1065 [0.0909, 0.1231] | 228,354 | 24,322 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 1,526 (0.7%) | 2,475 (1.2%) | 2,841 (1.3%) | 1,432 (0.7%) | 205,426 (96.1%) |
| `FRG` | 1,103 (0.5%) | — | 468 (0.2%) | 1,443 (0.6%) | 4,316 (1.8%) | 226,271 (96.9%) |
| `SAG.PB` | 145 (0.1%) | 286 (0.1%) | — | 2,605 (1.2%) | 512 (0.2%) | 210,375 (98.3%) |
| `SAG.PBM` | 1,642 (0.8%) | 2,602 (1.2%) | 2,251 (1.0%) | — | 4,165 (1.9%) | 203,893 (95.0%) |
| `ANY_ATTACK` | — | — | — | — | 2,843 (1.4%) | 201,189 (98.6%) |

### At 1000 alerts per 10,000 messages (budget 0.1)

| Target | threshold(s) | achieved alert rate | recall [95% CI] | precision [95% CI] | alerts | true |
|---|---|---:|---|---|---:|---:|
| `SAG.DB` | 0.000534857562, 0.00054540613, 0.000689437526, ... | 9.9602% | 0.9964 [0.9927, 0.9989] | 0.0219 [0.0163, 0.0279] | 2,313,420 | 50,597 |
| `FRG` | 0.00850817947, 0.00901722664, 0.00955643671, ... | 9.9731% | 0.6598 [0.5763, 0.7356] | 0.0182 [0.0128, 0.0239] | 2,316,394 | 42,209 |
| `SAG.PB` | 0.00757402756, 0.00772233139, 0.00787351607 | 9.8675% | 0.6465 [0.6062, 0.6877] | 0.0132 [0.0109, 0.0159] | 2,291,877 | 30,360 |
| `SAG.PBM` | 0.00802763663, 0.00834491071, 0.00884427658, ... | 9.8337% | 0.6928 [0.6325, 0.7472] | 0.0166 [0.0124, 0.0213] | 2,284,023 | 37,986 |
| `ANY_ATTACK` | 0.0236024165, 0.0240569577, 0.0245200327, ... | 9.8640% | 0.5488 [0.5115, 0.5888] | 0.0519 [0.0460, 0.0579] | 2,291,064 | 118,835 |

Where the false alarms come from, at this budget:

| Target | `SAG.DB` | `FRG` | `SAG.PB` | `SAG.PBM` | `benign_degradation` | `normal` |
|---|---|---|---|---|---|---|
| `SAG.DB` | — | 13,713 (0.6%) | 11,949 (0.5%) | 26,510 (1.2%) | 26,902 (1.2%) | 2,183,749 (96.5%) |
| `FRG` | 13,695 (0.6%) | — | 3,601 (0.2%) | 10,830 (0.5%) | 40,648 (1.8%) | 2,205,411 (97.0%) |
| `SAG.PB` | 20,842 (0.9%) | 3,324 (0.1%) | — | 14,028 (0.6%) | 5,567 (0.2%) | 2,217,756 (98.1%) |
| `SAG.PBM` | 29,046 (1.3%) | 12,875 (0.6%) | 14,043 (0.6%) | — | 16,450 (0.7%) | 2,173,623 (96.8%) |
| `ANY_ATTACK` | — | — | — | — | 22,358 (1.0%) | 2,149,871 (99.0%) |

## Paired comparison

Each replicate draws one set of runs and scores **both** configurations
on it, so the run-to-run variation they share cancels instead of being
counted twice. Two marginal intervals overlapping does not mean two
configurations are indistinguishable; this is the table that settles it.
Each configuration keeps its own cross-fold thresholds - the question is
which is better when each is operated properly, not which wins at a
threshold borrowed from the other.

### `d5-xgboost-none` (A) vs `d5-xgboost-downsample` (B)

| Target | B - A average precision | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0027 | [-0.0119, +0.0060] | no - indistinguishable |
| `FRG` | -0.0046 | [-0.0070, -0.0021] | **yes** |
| `SAG.PB` | -0.0044 | [-0.0090, -0.0013] | **yes** |
| `SAG.PBM` | -0.0051 | [-0.0076, -0.0030] | **yes** |
| `ANY_ATTACK` | -0.0100 | [-0.0135, -0.0069] | **yes** |

B - A recall at 1 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0045 | [-0.0020, +0.0133] | no |
| `FRG` | -0.0028 | [-0.0035, -0.0021] | **yes** |
| `SAG.PB` | -0.0063 | [-0.0109, -0.0023] | **yes** |
| `SAG.PBM` | -0.0019 | [-0.0046, +0.0002] | no |
| `ANY_ATTACK` | +0.0007 | [-0.0008, +0.0030] | no |

B - A recall at 10 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0030 | [-0.0184, +0.0298] | no |
| `FRG` | -0.0123 | [-0.0158, -0.0086] | **yes** |
| `SAG.PB` | -0.0144 | [-0.0223, -0.0070] | **yes** |
| `SAG.PBM` | -0.0135 | [-0.0193, -0.0086] | **yes** |
| `ANY_ATTACK` | -0.0031 | [-0.0087, +0.0047] | no |

B - A recall at 100 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | -0.0122 | [-0.0666, +0.0385] | no |
| `FRG` | -0.0215 | [-0.0419, +0.0017] | no |
| `SAG.PB` | -0.0120 | [-0.0237, -0.0005] | **yes** |
| `SAG.PBM` | -0.0254 | [-0.0346, -0.0167] | **yes** |
| `ANY_ATTACK` | -0.0214 | [-0.0328, -0.0090] | **yes** |

B - A recall at 1000 alert(s) per 10,000 messages:

| Target | B - A recall | 95% CI | separates? |
|---|---:|---|---|
| `SAG.DB` | +0.0017 | [-0.0003, +0.0036] | no |
| `FRG` | -0.0146 | [-0.0323, +0.0043] | no |
| `SAG.PB` | -0.0168 | [-0.0305, -0.0020] | **yes** |
| `SAG.PBM` | -0.0056 | [-0.0166, +0.0056] | no |
| `ANY_ATTACK` | -0.0137 | [-0.0221, -0.0049] | **yes** |

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

