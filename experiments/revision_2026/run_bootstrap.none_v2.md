# Run-level bootstrap intervals

- Generated: 2026-09-16 14:53:18 UTC
- Replicates: 2,000, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**
  Rows inside a run are correlated by construction, so a row-level bootstrap
  would report intervals far narrower than the evidence supports.

An interval here answers: *if the run matrix had drawn a different set of
runs from the same generator, how much would this number move?* It does not
capture uncertainty from the generator's design itself (attack prevalence,
loss rates and burst sizes are fixed by the matrix, not sampled).

## `v2-decision-tree-none`

- model: `decision-tree` | balance: `none` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.8421 [0.8176, 0.8645] | 0.8146 [0.7472, 0.8647] | 0.8281 [0.7887, 0.8577] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.6590 [0.6496, 0.6686] | 0.7058 [0.5894, 0.8081] | 0.6816 [0.6215, 0.7258] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.6741 [0.6431, 0.7070] | 0.8333 [0.7722, 0.8765] | 0.7453 [0.7109, 0.7717] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.1819 [0.1560, 0.2063] | 0.6562 [0.5494, 0.7388] | 0.2848 [0.2487, 0.3184] |
| `benign_degradation` | 85 | 270,642 | 0.5617 [0.5056, 0.6183] | 0.9099 [0.8694, 0.9388] | 0.6946 [0.6475, 0.7380] |
| `normal` | 245 | 10,570,307 | 0.9987 [0.9981, 0.9991] | 0.9843 [0.9789, 0.9887] | 0.9914 [0.9886, 0.9938] |

- macro F1: 0.7043 [0.6876, 0.7174]
- accuracy (micro): 0.9799 [0.9739, 0.9848]

## `v2-xgboost-none`

- model: `xgboost` | balance: `none` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.8828 [0.8606, 0.9021] | 0.8298 [0.7652, 0.8782] | 0.8555 [0.8175, 0.8837] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.6529 [0.6217, 0.6814] | 0.6634 [0.5460, 0.7626] | 0.6581 [0.5930, 0.7073] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.7272 [0.6953, 0.7592] | 0.8407 [0.7793, 0.8830] | 0.7798 [0.7447, 0.8069] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.2721 [0.2503, 0.2949] | 0.7064 [0.6142, 0.7788] | 0.3929 [0.3627, 0.4198] |
| `benign_degradation` | 85 | 270,642 | 0.5768 [0.5215, 0.6365] | 0.9134 [0.8779, 0.9383] | 0.7071 [0.6597, 0.7495] |
| `normal` | 245 | 10,570,307 | 0.9992 [0.9988, 0.9994] | 0.9863 [0.9813, 0.9904] | 0.9927 [0.9901, 0.9948] |

- macro F1: 0.7310 [0.7141, 0.7451]
- accuracy (micro): 0.9815 [0.9760, 0.9861]

## `v2-random-forest-none-cap4m`

- model: `random-forest` | balance: `none` | train cap: 4,000,000 | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.8415 [0.8173, 0.8636] | 0.8186 [0.7539, 0.8670] | 0.8299 [0.7913, 0.8585] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.5954 [0.5365, 0.6490] | 0.6408 [0.5282, 0.7402] | 0.6173 [0.5446, 0.6723] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.7296 [0.6921, 0.7640] | 0.8128 [0.7523, 0.8549] | 0.7690 [0.7327, 0.7969] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.3965 [0.3773, 0.4173] | 0.6151 [0.5431, 0.6729] | 0.4822 [0.4526, 0.5066] |
| `benign_degradation` | 85 | 270,642 | 0.6110 [0.5542, 0.6689] | 0.8818 [0.8344, 0.9163] | 0.7218 [0.6766, 0.7626] |
| `normal` | 245 | 10,570,307 | 0.9981 [0.9977, 0.9984] | 0.9878 [0.9832, 0.9915] | 0.9929 [0.9905, 0.9949] |

- macro F1: 0.7355 [0.7183, 0.7494]
- accuracy (micro): 0.9815 [0.9760, 0.9860]

## `v2-logistic-regression-none`

- model: `logistic-regression` | balance: `none` | train cap: — | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.5953 [0.5372, 0.6528] | 0.5624 [0.4688, 0.6434] | 0.5784 [0.5109, 0.6340] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.3032 [0.2297, 0.3870] | 0.6826 [0.5751, 0.7617] | 0.4199 [0.3327, 0.5045] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0007 [0.0004, 0.0010] | 0.0369 [0.0183, 0.0633] | 0.0013 [0.0007, 0.0020] |
| `benign_degradation` | 85 | 270,642 | 0.1413 [0.0981, 0.2034] | 0.5340 [0.4365, 0.6239] | 0.2234 [0.1618, 0.2962] |
| `normal` | 245 | 10,570,307 | 0.9988 [0.9985, 0.9989] | 0.9676 [0.9571, 0.9762] | 0.9829 [0.9774, 0.9875] |

- macro F1: 0.3677 [0.3486, 0.3879]
- accuracy (micro): 0.9622 [0.9511, 0.9713]

## `v2-decision-tree-none-cap4m`

- model: `decision-tree` | balance: `none` | train cap: 4,000,000 | runs covered: **265**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 45 | 50,779 | 0.8370 [0.8115, 0.8603] | 0.8162 [0.7495, 0.8663] | 0.8265 [0.7870, 0.8561] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 45 | 63,963 | 0.6532 [0.6414, 0.6645] | 0.7074 [0.5906, 0.8107] | 0.6792 [0.6193, 0.7229] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.6663 [0.6346, 0.6993] | 0.8272 [0.7648, 0.8713] | 0.7381 [0.7032, 0.7647] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.1865 [0.1595, 0.2133] | 0.6554 [0.5519, 0.7380] | 0.2904 [0.2522, 0.3259] |
| `benign_degradation` | 85 | 270,642 | 0.5615 [0.5061, 0.6185] | 0.9100 [0.8689, 0.9387] | 0.6945 [0.6468, 0.7375] |
| `normal` | 245 | 10,570,307 | 0.9987 [0.9981, 0.9991] | 0.9842 [0.9788, 0.9886] | 0.9914 [0.9885, 0.9938] |

- macro F1: 0.7033 [0.6870, 0.7166]
- accuracy (micro): 0.9798 [0.9739, 0.9847]

## Paired comparison (macro F1)

Each replicate draws one set of runs and scores **both** models on it, so
the run-to-run variation the two share cancels instead of being counted
twice. Two marginal intervals overlapping does not mean two models are
indistinguishable, and this table is what actually settles it.

| A | B | B - A | 95% CI | separates? |
|---|---|---:|---|---|
| `v2-decision-tree-none` | `v2-xgboost-none` | +0.0267 | [+0.0208, +0.0322] | **yes** |
| `v2-decision-tree-none` | `v2-random-forest-none-cap4m` | +0.0312 | [+0.0219, +0.0395] | **yes** |
| `v2-decision-tree-none` | `v2-logistic-regression-none` | -0.3366 | [-0.3572, -0.3127] | **yes** |
| `v2-decision-tree-none` | `v2-decision-tree-none-cap4m` | -0.0010 | [-0.0029, +0.0011] | no - indistinguishable |
