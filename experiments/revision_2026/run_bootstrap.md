# Run-level bootstrap intervals

- Generated: 2026-09-13 06:56:46 UTC
- Replicates: 2,000, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**
  Rows inside a run are correlated by construction, so a row-level bootstrap
  would report intervals far narrower than the evidence supports.

An interval here answers: *if the run matrix had drawn a different set of
runs from the same generator, how much would this number move?* It does not
capture uncertainty from the generator's design itself (attack prevalence,
loss rates and burst sizes are fixed by the matrix, not sampled).

## `d3-xgboost-downsample`

- model: `xgboost` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.9024 [0.8661, 0.9351] | 0.0370 [0.0280, 0.0469] | 0.0712 [0.0543, 0.0893] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.6292 [0.5602, 0.6881] | 0.0126 [0.0089, 0.0166] | 0.0248 [0.0174, 0.0324] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.6765 [0.6346, 0.7187] | 0.0124 [0.0104, 0.0147] | 0.0243 [0.0206, 0.0287] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.5716 [0.5420, 0.6028] | 0.0112 [0.0084, 0.0143] | 0.0221 [0.0165, 0.0279] |
| `benign_degradation` | 85 | 270,680 | 0.6916 [0.5979, 0.7653] | 0.2801 [0.2145, 0.3358] | 0.3988 [0.3181, 0.4626] |
| `normal` | 265 | 22,739,305 | 0.5613 [0.5255, 0.5899] | 0.9981 [0.9974, 0.9986] | 0.7186 [0.6884, 0.7417] |

- macro F1: 0.2099 [0.1963, 0.2209]
- accuracy (micro): 0.5641 [0.5293, 0.5921]

## `d3-decision-tree-downsample`

- model: `decision-tree` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.9156 [0.8853, 0.9448] | 0.0348 [0.0261, 0.0443] | 0.0670 [0.0508, 0.0846] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.5718 [0.5041, 0.6303] | 0.0119 [0.0083, 0.0157] | 0.0233 [0.0163, 0.0305] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.6365 [0.5916, 0.6829] | 0.0090 [0.0076, 0.0107] | 0.0177 [0.0151, 0.0211] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.6026 [0.5757, 0.6302] | 0.0107 [0.0080, 0.0136] | 0.0210 [0.0157, 0.0266] |
| `benign_degradation` | 85 | 270,680 | 0.6319 [0.5440, 0.7017] | 0.3283 [0.2499, 0.3920] | 0.4321 [0.3440, 0.4995] |
| `normal` | 265 | 22,739,305 | 0.5211 [0.4920, 0.5433] | 0.9978 [0.9969, 0.9984] | 0.6847 [0.6588, 0.7037] |

- macro F1: 0.2076 [0.1933, 0.2193]
- accuracy (micro): 0.5238 [0.4959, 0.5456]

## `d3-random-forest-downsample`

- model: `random-forest` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.6606 [0.6139, 0.7039] | 0.0367 [0.0276, 0.0467] | 0.0696 [0.0528, 0.0873] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.6217 [0.5625, 0.6734] | 0.0121 [0.0086, 0.0159] | 0.0238 [0.0170, 0.0310] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.6626 [0.6190, 0.7041] | 0.0118 [0.0099, 0.0142] | 0.0233 [0.0194, 0.0278] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.6217 [0.5998, 0.6455] | 0.0144 [0.0108, 0.0184] | 0.0282 [0.0212, 0.0357] |
| `benign_degradation` | 85 | 270,680 | 0.7115 [0.6218, 0.7809] | 0.1587 [0.1114, 0.2056] | 0.2595 [0.1888, 0.3229] |
| `normal` | 265 | 22,739,305 | 0.5630 [0.5267, 0.5911] | 0.9978 [0.9971, 0.9983] | 0.7199 [0.6893, 0.7426] |

- macro F1: 0.1874 [0.1756, 0.1978]
- accuracy (micro): 0.5655 [0.5306, 0.5928]

## `d3-logistic-regression-downsample`

- model: `logistic-regression` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.7368 [0.6557, 0.8144] | 0.0195 [0.0145, 0.0250] | 0.0381 [0.0284, 0.0483] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.3322 [0.3048, 0.3560] | 0.0079 [0.0056, 0.0104] | 0.0155 [0.0109, 0.0202] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.5378 [0.4772, 0.6009] | 0.0064 [0.0054, 0.0078] | 0.0127 [0.0107, 0.0153] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.5383 [0.4991, 0.5804] | 0.0127 [0.0095, 0.0162] | 0.0248 [0.0186, 0.0315] |
| `benign_degradation` | 85 | 270,680 | 0.5093 [0.4458, 0.5667] | 0.0598 [0.0399, 0.0824] | 0.1070 [0.0736, 0.1433] |
| `normal` | 265 | 22,739,305 | 0.4420 [0.4067, 0.4698] | 0.9976 [0.9968, 0.9982] | 0.6126 [0.5777, 0.6388] |

- macro F1: 0.1351 [0.1284, 0.1415]
- accuracy (micro): 0.4436 [0.4093, 0.4707]

## `d3-random-forest-none-cap4m`

- model: `random-forest` | balance: `none` | train cap: 4,000,000 | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.0116 [0.0064, 0.0187] | 0.0362 [0.0195, 0.0580] | 0.0176 [0.0098, 0.0279] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.0406 [0.0294, 0.0525] | 0.1543 [0.1051, 0.2062] | 0.0643 [0.0471, 0.0812] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0357 [0.0226, 0.0497] | 0.1652 [0.1097, 0.2204] | 0.0587 [0.0377, 0.0808] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0565 [0.0509, 0.0619] | 0.2202 [0.1835, 0.2556] | 0.0899 [0.0814, 0.0983] |
| `benign_degradation` | 85 | 270,680 | 0.5888 [0.4926, 0.6708] | 0.8161 [0.7504, 0.8748] | 0.6841 [0.6008, 0.7538] |
| `normal` | 265 | 22,739,305 | 0.9964 [0.9954, 0.9972] | 0.9863 [0.9833, 0.9886] | 0.9913 [0.9894, 0.9928] |

- macro F1: 0.3176 [0.3026, 0.3305]
- accuracy (micro): 0.9827 [0.9790, 0.9856]

## `d3-xgboost-none`

- model: `xgboost` | balance: `none` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.0000 [0.0000, 0.0001] | 0.0039 [0.0000, 0.0333] | 0.0000 [0.0000, 0.0001] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `benign_degradation` | 85 | 270,680 | 0.6003 [0.5056, 0.6824] | 0.8438 [0.7754, 0.9090] | 0.7015 [0.6175, 0.7741] |
| `normal` | 265 | 22,739,305 | 0.9987 [0.9977, 0.9994] | 0.9859 [0.9828, 0.9882] | 0.9922 [0.9903, 0.9936] |

- macro F1: 0.2823 [0.2682, 0.2946]
- accuracy (micro): 0.9847 [0.9810, 0.9874]

## `d3-decision-tree-none`

- model: `decision-tree` | balance: `none` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.0000 [0.0000, 0.0000] | 0.1250 [0.0000, 0.4304] | 0.0000 [0.0000, 0.0001] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0000 [0.0000, 0.0001] | 0.2857 [0.0000, 0.5000] | 0.0001 [0.0000, 0.0003] |
| `benign_degradation` | 85 | 270,680 | 0.5823 [0.4776, 0.6699] | 0.8249 [0.7489, 0.9001] | 0.6827 [0.5928, 0.7593] |
| `normal` | 265 | 22,739,305 | 0.9985 [0.9974, 0.9994] | 0.9857 [0.9826, 0.9880] | 0.9921 [0.9901, 0.9935] |

- macro F1: 0.2791 [0.2640, 0.2920]
- accuracy (micro): 0.9844 [0.9806, 0.9872]

## `d3-decision-tree-none-cap4m`

- model: `decision-tree` | balance: `none` | train cap: 4,000,000 | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.0000 [0.0000, 0.0001] | 0.0667 [0.0000, 0.1622] | 0.0001 [0.0000, 0.0002] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0000 [0.0000, 0.0001] | 0.0556 [0.0000, 0.1852] | 0.0001 [0.0000, 0.0002] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0000 [0.0000, 0.0001] | 0.0233 [0.0000, 0.0328] | 0.0000 [0.0000, 0.0001] |
| `benign_degradation` | 85 | 270,680 | 0.5808 [0.4769, 0.6678] | 0.8234 [0.7475, 0.8964] | 0.6811 [0.5924, 0.7570] |
| `normal` | 265 | 22,739,305 | 0.9985 [0.9975, 0.9993] | 0.9857 [0.9826, 0.9880] | 0.9921 [0.9901, 0.9935] |

- macro F1: 0.2789 [0.2641, 0.2917]
- accuracy (micro): 0.9843 [0.9806, 0.9871]

## `d3-logistic-regression-none`

- model: `logistic-regression` | balance: `none` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `benign_degradation` | 85 | 270,680 | 0.0000 [0.0000, 0.0000] | 0.0027 [0.0000, 0.0094] | 0.0000 [0.0000, 0.0000] |
| `normal` | 265 | 22,739,305 | 1.0000 [1.0000, 1.0000] | 0.9790 [0.9734, 0.9833] | 0.9894 [0.9865, 0.9916] |

- macro F1: 0.1649 [0.1644, 0.1653]
- accuracy (micro): 0.9790 [0.9734, 0.9833]

## `grouped-validation-full-smote`

- model: `decision-tree` | balance: `smote` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,782 | 0.0001 [0.0000, 0.0001] | 0.0007 [0.0000, 0.0017] | 0.0001 [0.0000, 0.0002] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,976 | 0.0000 [0.0000, 0.0001] | 0.0014 [0.0000, 0.0058] | 0.0000 [0.0000, 0.0001] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0012 [0.0004, 0.0023] | 0.0053 [0.0018, 0.0108] | 0.0019 [0.0007, 0.0035] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `benign_degradation` | 85 | 270,680 | 0.5131 [0.4187, 0.5931] | 0.8169 [0.7384, 0.8970] | 0.6303 [0.5412, 0.7064] |
| `normal` | 265 | 22,739,305 | 0.9980 [0.9970, 0.9988] | 0.9850 [0.9817, 0.9875] | 0.9915 [0.9894, 0.9929] |

- macro F1: 0.2706 [0.2558, 0.2834]
- accuracy (micro): 0.9831 [0.9791, 0.9860]
