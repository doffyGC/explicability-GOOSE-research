# Run-level bootstrap intervals

- Generated: 2026-09-13 07:05:15 UTC
- Replicates: 2,000, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**
  Rows inside a run are correlated by construction, so a row-level bootstrap
  would report intervals far narrower than the evidence supports.

An interval here answers: *if the run matrix had drawn a different set of
runs from the same generator, how much would this number move?* It does not
capture uncertainty from the generator's design itself (attack prevalence,
loss rates and burst sizes are fixed by the matrix, not sampled).

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

## Paired comparison (macro F1)

Each replicate draws one set of runs and scores **both** models on it, so
the run-to-run variation the two share cancels instead of being counted
twice. Two marginal intervals overlapping does not mean two models are
indistinguishable, and this table is what actually settles it.

| A | B | B - A | 95% CI | separates? |
|---|---|---:|---|---|
| `d3-decision-tree-downsample` | `d3-xgboost-downsample` | +0.0023 | [+0.0000, +0.0045] | **yes** |
| `d3-decision-tree-downsample` | `d3-random-forest-downsample` | -0.0202 | [-0.0233, -0.0165] | **yes** |
| `d3-decision-tree-downsample` | `d3-logistic-regression-downsample` | -0.0725 | [-0.0789, -0.0630] | **yes** |
