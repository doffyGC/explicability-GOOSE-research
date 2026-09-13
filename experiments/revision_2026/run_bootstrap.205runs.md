# Run-level bootstrap intervals

- Generated: 2026-09-13 03:09:48 UTC
- Replicates: 2,000, percentile interval at 95%, seed 42
- **Resampling unit: `split_group` (one ERENO run), not rows and not folds.**
  Rows inside a run are correlated by construction, so a row-level bootstrap
  would report intervals far narrower than the evidence supports.

An interval here answers: *if the run matrix had drawn a different set of
runs from the same generator, how much would this number move?* It does not
capture uncertainty from the generator's design itself (attack prevalence,
loss rates and burst sizes are fixed by the matrix, not sampled).

## `d3-xgboost-downsample`

- model: `xgboost` | balance: `downsample` | train cap: — | runs covered: **205**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 15 | 17,094 | 0.8975 [0.8358, 0.9542] | 0.0151 [0.0081, 0.0231] | 0.0296 [0.0161, 0.0451] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 15 | 21,436 | 0.5971 [0.4951, 0.6771] | 0.0046 [0.0023, 0.0074] | 0.0091 [0.0046, 0.0146] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.6651 [0.6219, 0.7072] | 0.0127 [0.0107, 0.0152] | 0.0249 [0.0211, 0.0297] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.5640 [0.5355, 0.5956] | 0.0123 [0.0092, 0.0160] | 0.0241 [0.0181, 0.0310] |
| `benign_degradation` | 85 | 270,680 | 0.6843 [0.5820, 0.7585] | 0.2847 [0.2170, 0.3424] | 0.4021 [0.3185, 0.4680] |
| `normal` | 205 | 20,385,924 | 0.5570 [0.5153, 0.5872] | 0.9985 [0.9978, 0.9989] | 0.7151 [0.6796, 0.7395] |

- macro F1: 0.2008 [0.1860, 0.2120]
- accuracy (micro): 0.5592 [0.5187, 0.5888]

> **Thin classes:** `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE`, `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` carry fewer than 20 independent runs, so their
> intervals are wide, and coarse - they move in steps of roughly one
> run's worth of the metric. That is thin evidence, not an unstable
> model. Narrowing them needs more runs of those variants, not more rows
> per run.

## `d3-random-forest-none-cap4m`

- model: `random-forest` | balance: `none` | train cap: 4,000,000 | runs covered: **205**

| Class | runs carrying it | rows | recall [95% CI] | precision [95% CI] | F1 [95% CI] |
|---|---:|---:|---|---|---|
| `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE` | 15 | 17,094 | 0.0009 [0.0004, 0.0015] | 0.0032 [0.0011, 0.0061] | 0.0014 [0.0005, 0.0023] |
| `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` | 15 | 21,436 | 0.0014 [0.0008, 0.0021] | 0.0122 [0.0051, 0.0241] | 0.0025 [0.0015, 0.0037] |
| `RANDOMIC_BURST_ORIENTEDGRAYHOLE` | 45 | 46,959 | 0.0379 [0.0238, 0.0530] | 0.1689 [0.1108, 0.2294] | 0.0619 [0.0393, 0.0863] |
| `RANDOMIC_MESSAGE_ORIENTEDGRAYHOLE` | 45 | 54,828 | 0.0570 [0.0495, 0.0647] | 0.2209 [0.1821, 0.2590] | 0.0906 [0.0789, 0.1012] |
| `benign_degradation` | 85 | 270,680 | 0.5901 [0.4851, 0.6714] | 0.8130 [0.7470, 0.8692] | 0.6838 [0.5950, 0.7543] |
| `normal` | 205 | 20,385,924 | 0.9970 [0.9960, 0.9978] | 0.9882 [0.9853, 0.9903] | 0.9926 [0.9907, 0.9939] |

- macro F1: 0.3055 [0.2900, 0.3180]
- accuracy (micro): 0.9852 [0.9816, 0.9879]

> **Thin classes:** `DETERMINISTIC_BURST_ORIENTEDGRAYHOLE`, `FULLY_RANDOMIZED_ORIENTEDGRAYHOLE` carry fewer than 20 independent runs, so their
> intervals are wide, and coarse - they move in steps of roughly one
> run's worth of the metric. That is thin evidence, not an unstable
> model. Narrowing them needs more runs of those variants, not more rows
> per run.
