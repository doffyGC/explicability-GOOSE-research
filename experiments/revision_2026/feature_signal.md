# Where the attack signal is

- Generated: 2026-09-13 15:13:26 UTC
- Run: `d5-xgboost-none` | dataset rows: 23,226,530 | attack prevalence: 0.9323%

Average precision on one axis. **Chance is the prevalence, 0.0093** - a
single feature scoring near it carries no marginal signal at all.

| Score | AP | x chance |
|---|---:|---:|
| `[model] xgboost/none` | 0.0724 | 7.8x |
| `timeFromLastChange` | 0.0168 | 1.8x |
| `timestampDiff` | 0.0103 | 1.1x |
| `stDiff` | 0.0102 | 1.1x |
| `[rule] stDiff != 0` | 0.0101 | 1.1x |
| `cbStatusDiff` | 0.0099 | 1.1x |
| `sqDiff` | 0.0097 | 1.0x |
| `tDiff` | 0.0095 | 1.0x |
| `gooseLengthDiff` | 0.0093 | 1.0x |
| `apduSizeDiff` | 0.0093 | 1.0x |
| `frameLengthDiff` | 0.0093 | 1.0x |
| `[rule] sqDiff != 0` | 0.0091 | 1.0x |
| `delay` | 0.0090 | 1.0x |

