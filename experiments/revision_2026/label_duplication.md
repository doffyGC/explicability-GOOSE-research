# Label/feature duplication audit

- Generated: 2026-09-13 15:07:35 UTC
- Source: `data/runs`

| Quantity | Value |
|---|---:|
| runs audited | 265 |
| rows | 23,226,795 |
| rows sharing a message key with another row | 21,679,174 (93.3%) |
| attack rows | 216,547 |
| **attack rows with a content-identical non-attack row** | **216,547 (100.00%)** |
| attack rows identical on *every* model feature (irreducible) | 8,616 (3.98%) |
| **rows whose content is repeated under a different label** (any label) | **562,788 (2.42%)** |
| same, identical on every model feature | 19,000 |

`content` excludes the eight inter-message delta columns, which differ on a
duplicate by construction; `full` includes them, so a full twin cannot be
separated by any classifier reading this feature matrix.

## What differs between two copies of the same message

Diagnostic from `BENIGN_CONGESTION_LOSS-l15-s20260101` (7,666 duplicated keys). A pair differing *only* in the
delta columns and `class` is one message written twice, not two messages.

| Column | keys where the copies differ |
|---|---:|
| `timestampDiff` | 7,106 (92.7%) |
| `sqDiff` | 7,092 (92.5%) |
| `stDiff` | 1,293 (16.9%) |
| `tDiff` | 1,293 (16.9%) |
| `class` | 1,126 (14.7%) |
| `cbStatusDiff` | 1,047 (13.7%) |

