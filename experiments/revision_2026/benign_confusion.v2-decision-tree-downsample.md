# Benign-degradation confusion report - Gray-GOOSE (card C)

- Generated: 2026-09-16 14:56:10 UTC
- Source run: `C:\Users\ResTIC16\Desktop\Pessoal\Pesquisa\Cybersegurança\explicability-GOOSE-research\data\runs\gray-GOOSE-runs-prepared.parquet`
- Run status: `full_grouped_run`
- Protocol: `stratified-group-kfold`
- Model: `decision-tree`
- Dataset re-verified against: `data/runs/gray-GOOSE-runs-prepared.parquet`

## 1. Confusion matrix

| true \ predicted | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM | total |
|---|---|---|---|---|---|---|---|
| normal | 8,010,739 | 1,357,774 | 33,469 | 101,786 | 90,528 | 976,011 | 10,570,307 |
| benign_degradation | 1,077 | 224,073 | 6,575 | 22,871 | 3,208 | 12,838 | 270,642 |
| SAG.DB | 0 | 34 | 48,674 | 0 | 2,017 | 54 | 50,779 |
| FRG | 591 | 1,138 | 1,106 | 46,356 | 508 | 14,264 | 63,963 |
| SAG.PB | 272 | 204 | 7,935 | 258 | 37,229 | 1,061 | 46,959 |
| SAG.PBM | 547 | 286 | 2,435 | 3,973 | 2,334 | 45,253 | 54,828 |

## 2. Per-mode outcome breakdown

What each impairment mode's `benign_degradation` rows actually get predicted
as, across every class present in this run.

| mode | n | normal | benign_degradation | SAG.DB | FRG | SAG.PB | SAG.PBM |
|---|---|---|---|---|---|---|---|
| CONGESTION_LOSS | 21,428 | 123 (0.6%) | 299 (1.4%) | 346 (1.6%) | 15,712 (73.3%) | 199 (0.9%) | 4,749 (22.2%) |
| DELAY | 89,980 | 41 (0.0%) | 88,210 (98.0%) | 410 (0.5%) | 876 (1.0%) | 17 (0.0%) | 426 (0.5%) |
| DUPLICATION | 13,519 | 113 (0.8%) | 13,168 (97.4%) | 0 (0.0%) | 0 (0.0%) | 13 (0.1%) | 225 (1.7%) |
| JITTER | 89,980 | 162 (0.2%) | 87,078 (96.8%) | 364 (0.4%) | 1,008 (1.1%) | 33 (0.0%) | 1,335 (1.5%) |
| LINK_FLAP | 20,205 | 230 (1.1%) | 12,445 (61.6%) | 2,451 (12.1%) | 1,966 (9.7%) | 1,485 (7.3%) | 1,628 (8.1%) |
| QUEUE_OVERLOAD_BURST | 23,200 | 200 (0.9%) | 13,686 (59.0%) | 2,894 (12.5%) | 3,033 (13.1%) | 1,443 (6.2%) | 1,944 (8.4%) |
| REORDERING | 12,330 | 208 (1.7%) | 9,187 (74.5%) | 110 (0.9%) | 276 (2.2%) | 18 (0.1%) | 2,531 (20.5%) |

## 3. False-positive rate and alert burden by mode

`attack_fpr`: fraction of a slice's rows predicted as one of the four attack
classes - the specific confusion each mechanism is paired to falsify
(benign_controls.md SS3). `alert_rate`: fraction predicted as anything other
than `normal` (`benign_degradation` included) - the broader "does this traffic
look unusual at all" question.

| slice | n | attack_fpr | alert_rate |
|---|---:|---:|---:|
| normal (ideal, impairment_mode=NONE) | 9,649,593 | 10.46% | 21.40% |
| normal (baseline messages inside a benign-impairment run) | 920,714 | 20.87% | 53.76% |
| benign_degradation: CONGESTION_LOSS | 21,428 | 98.03% | 99.43% |
| benign_degradation: DELAY | 89,980 | 1.92% | 99.95% |
| benign_degradation: DUPLICATION | 13,519 | 1.76% | 99.16% |
| benign_degradation: JITTER | 89,980 | 3.05% | 99.82% |
| benign_degradation: LINK_FLAP | 20,205 | 37.27% | 98.86% |
| benign_degradation: QUEUE_OVERLOAD_BURST | 23,200 | 40.15% | 99.14% |
| benign_degradation: REORDERING | 12,330 | 23.80% | 98.31% |

> Highest attack_fpr: **CONGESTION_LOSS** (98.03%). A high
> attack_fpr on a mode means the model is learning "gap/anomaly in traffic" as a
> proxy for "attack" rather than the attack's actual signature - exactly the
> failure mode this card exists to surface (see CLAUDE.md, "Known baseline issues
> driving the revision").

