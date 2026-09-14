# Label/feature duplication audit

- Generated: 2026-09-14 00:23:19 UTC
- Source: `data/runs`

| Quantity | Value |
|---|---:|
| runs audited | 265 |
| rows | 11,057,743 |
| rows sharing a message key with another row | 0 (0.0%) |
| attack rows | 216,542 |
| **attack rows with a content-identical non-attack row** | **0 (0.00%)** |
| attack rows identical on *every* model feature (irreducible) | 0 (0.00%) |
| **rows whose content is repeated under a different label** (any label) | **0 (0.00%)** |
| same, identical on every model feature | 0 |

`content` excludes the eight inter-message delta columns, which differ on a
duplicate by construction; `full` includes them, so a full twin cannot be
separated by any classifier reading this feature matrix.

