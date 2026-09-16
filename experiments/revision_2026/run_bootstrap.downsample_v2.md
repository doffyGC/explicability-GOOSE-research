# Run-level bootstrap intervals

- Generated: 2026-09-16 14:54:09 UTC
- Replicates: 2,000, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**
  Rows inside a run are correlated by construction, so a row-level bootstrap
  would report intervals far narrower than the evidence supports.

An interval here answers: *if the run matrix had drawn a different set of
runs from the same generator, how much would this number move?* It does not
capture uncertainty from the generator's design itself (attack prevalence,
loss rates and burst sizes are fixed by the matrix, not sampled).

## `v2-decision-tree-downsample`

- model: `decision-tree` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.9585 [0.9417, 0.9737] | 0.4858 [0.4007, 0.5582] | 0.6448 [0.5659, 0.7054] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.7247 [0.7060, 0.7426] | 0.2645 [0.2023, 0.3211] | 0.3876 [0.3156, 0.4462] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.7928 [0.7474, 0.8344] | 0.2741 [0.2354, 0.3147] | 0.4074 [0.3644, 0.4483] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.8254 [0.7941, 0.8575] | 0.0431 [0.0317, 0.0557] | 0.0820 [0.0611, 0.1042] |
| `benign_degradation` | 85 | 270,642 | 0.8279 [0.7577, 0.8721] | 0.1415 [0.0984, 0.1867] | 0.2417 [0.1746, 0.3071] |
| `normal` | 245 | 10,570,307 | 0.7579 [0.7257, 0.7829] | 0.9997 [0.9996, 0.9998] | 0.8621 [0.8409, 0.8782] |

- macro F1: 0.4376 [0.4175, 0.4543]
- accuracy (micro): 0.7608 [0.7309, 0.7845]

## `v2-xgboost-downsample`

- model: `xgboost` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.9583 [0.9360, 0.9767] | 0.5187 [0.4331, 0.5913] | 0.6731 [0.5970, 0.7312] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.7780 [0.7503, 0.8050] | 0.2733 [0.2131, 0.3296] | 0.4045 [0.3330, 0.4649] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.8339 [0.7986, 0.8670] | 0.2962 [0.2551, 0.3395] | 0.4372 [0.3920, 0.4800] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.8197 [0.7888, 0.8519] | 0.0547 [0.0405, 0.0698] | 0.1025 [0.0772, 0.1288] |
| `benign_degradation` | 85 | 270,642 | 0.8618 [0.8020, 0.9028] | 0.1528 [0.1078, 0.1989] | 0.2595 [0.1913, 0.3258] |
| `normal` | 245 | 10,570,307 | 0.7850 [0.7551, 0.8080] | 0.9996 [0.9994, 0.9998] | 0.8794 [0.8603, 0.8937] |

- macro F1: 0.4594 [0.4382, 0.4764]
- accuracy (micro): 0.7880 [0.7599, 0.8098]

## `v2-random-forest-downsample`

- model: `random-forest` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.8962 [0.8657, 0.9243] | 0.5716 [0.4869, 0.6422] | 0.6980 [0.6304, 0.7498] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.7814 [0.7496, 0.8130] | 0.2161 [0.1645, 0.2651] | 0.3385 [0.2708, 0.3957] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.8345 [0.7989, 0.8680] | 0.2381 [0.2019, 0.2790] | 0.3705 [0.3267, 0.4160] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.8161 [0.7944, 0.8380] | 0.0646 [0.0477, 0.0818] | 0.1196 [0.0902, 0.1490] |
| `benign_degradation` | 85 | 270,642 | 0.8479 [0.7906, 0.8879] | 0.1651 [0.1175, 0.2138] | 0.2764 [0.2053, 0.3446] |
| `normal` | 245 | 10,570,307 | 0.8032 [0.7761, 0.8240] | 0.9989 [0.9985, 0.9993] | 0.8904 [0.8734, 0.9033] |

- macro F1: 0.4489 [0.4283, 0.4659]
- accuracy (micro): 0.8048 [0.7793, 0.8245]

## `v2-logistic-regression-downsample`

- model: `logistic-regression` | balance: `downsample` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.9266 [0.8978, 0.9559] | 0.4129 [0.3342, 0.4857] | 0.5713 [0.4908, 0.6374] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.3769 [0.3623, 0.3916] | 0.0290 [0.0209, 0.0373] | 0.0538 [0.0395, 0.0677] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.7057 [0.6575, 0.7527] | 0.1124 [0.0907, 0.1457] | 0.1939 [0.1614, 0.2388] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.7529 [0.7120, 0.7936] | 0.0351 [0.0262, 0.0440] | 0.0670 [0.0506, 0.0832] |
| `benign_degradation` | 85 | 270,642 | 0.4718 [0.4136, 0.5135] | 0.1043 [0.0682, 0.1414] | 0.1709 [0.1174, 0.2210] |
| `normal` | 245 | 10,570,307 | 0.6992 [0.6603, 0.7297] | 0.9966 [0.9950, 0.9978] | 0.8219 [0.7938, 0.8430] |

- macro F1: 0.3131 [0.2983, 0.3265]
- accuracy (micro): 0.6931 [0.6543, 0.7240]

## Paired comparison (macro F1)

Each replicate draws one set of runs and scores **both** models on it, so
the run-to-run variation the two share cancels instead of being counted
twice. Two marginal intervals overlapping does not mean two models are
indistinguishable, and this table is what actually settles it.

| A | B | B - A | 95% CI | separates? |
|---|---|---:|---|---|
| `v2-decision-tree-downsample` | `v2-xgboost-downsample` | +0.0218 | [+0.0184, +0.0252] | **yes** |
| `v2-decision-tree-downsample` | `v2-random-forest-downsample` | +0.0113 | [+0.0064, +0.0165] | **yes** |
| `v2-decision-tree-downsample` | `v2-logistic-regression-downsample` | -0.1245 | [-0.1328, -0.1136] | **yes** |
