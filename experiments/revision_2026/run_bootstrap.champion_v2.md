# Run-level bootstrap intervals

- Generated: 2026-09-16 14:54:48 UTC
- Replicates: 2,000, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**
  Rows inside a run are correlated by construction, so a row-level bootstrap
  would report intervals far narrower than the evidence supports.

An interval here answers: *if the run matrix had drawn a different set of
runs from the same generator, how much would this number move?* It does not
capture uncertainty from the generator's design itself (attack prevalence,
loss rates and burst sizes are fixed by the matrix, not sampled).

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

## Paired comparison (macro F1)

Each replicate draws one set of runs and scores **both** models on it, so
the run-to-run variation the two share cancels instead of being counted
twice. Two marginal intervals overlapping does not mean two models are
indistinguishable, and this table is what actually settles it.

| A | B | B - A | 95% CI | separates? |
|---|---|---:|---|---|
| `v2-xgboost-none` | `v2-random-forest-none-cap4m` | +0.0045 | [-0.0036, +0.0117] | no - indistinguishable |
