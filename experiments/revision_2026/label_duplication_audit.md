# The label is not a function of the features (2026-09-13)

**Status: blocking. Every attack-detection number in cards D and E is measured
on a dataset where the label cannot be predicted from the features, because
the same message appears twice under two different labels.**

Found while designing card D.2's rule-based baseline: the `sqDiff`/`stDiff`
conventions did not behave like GOOSE sequence deltas, and the reason turned
out not to be a convention at all.

## 1. The finding

**Every attack row in the regenerated 265-run pool has a row labelled
`normal`, in the same run, whose content features are bit-identical.**

| Class | attack rows | with a content-identical `normal` row | identical on *every* model feature |
|---|---:|---:|---:|
| `SAG.DB` | 50,782 | **100.0%** | 0.0% |
| `FRG` | 63,976 | **100.0%** | 2.9% |
| `SAG.PB` | 46,959 | **100.0%** | 0.2% |
| `SAG.PBM` | 54,828 | **100.0%** | 12.2% |
| **total** | **216,545** | **100.0%** | **3.98%** |

"Content" is the 32 model features describing the message itself - every
protocol field, every SV electrical value, absolute timing. The eight
inter-message delta columns are excluded because a duplicate row necessarily
differs there; including them would hide the duplication behind the arithmetic
it causes.

The 3.98% column is the harder statement: those rows are identical on **all
40** model features and a non-attack row. No classifier reading this feature
matrix can separate them, at any threshold, ever.

Across the whole pool, on the raw ERENO CSVs
(`label_duplication.md`, 265 runs, 23,226,795 rows):

| Quantity | Value |
|---|---:|
| rows sharing a message key with another row | 21,679,174 (93.3%) |
| **attack rows with a content-identical non-attack row** | **216,547 / 216,547 (100.00%)** |
| attack rows identical on *every* model feature | 8,616 (3.98%) |
| rows carrying a label their content twin contradicts (any label) | 562,788 (2.42%) |
| same, identical on every model feature | 19,000 |

The row count cross-checks: 23,226,795 raw minus one predecessor-less first
row per trace (265) is the 23,226,530 the prepared dataset holds.

## 2. It is in the raw ERENO output, not in the pipeline

The defect is present in `data/runs/*.csv` as ERENO wrote them, before
`merge_runs.py`, `add_experiment_metadata.py` or `prepare_grouped_dataset.py`
touch anything.

    DETERMINISTIC_BURST-l100-b10-s20260101.csv
      58,132 rows, 36,000 distinct (StNum, SqNum, GooseTimestamp, t) keys
      1,276 attack rows, of which 1,276 (100.0%) share a key with a `normal` row

    RANDOMIC_MESSAGE-l15-b10-s20260101.csv
      52,461 rows, 27,000 keys
      1,348 attack rows, of which 1,348 (100.0%) share a key with a `normal` row

## 3. What the two copies are

Across all 65 raw columns, the copies of one message differ **only** here:

| Column | keys where the copies differ |
|---|---:|
| `timestampDiff` | 99.8% |
| `sqDiff` | 99.8% |
| `class` | 5.8% |
| `tDiff` | 4.7% |
| `stDiff` | 4.6% |
| `cbStatusDiff` | 2.4% |

Five of those six are inter-message deltas, which are relative to the
predecessor and so differ on a duplicate by construction. The sixth is the
label, and it differs on exactly the attacked messages.

Everything else matches: `ethSrc`, `ethDst`, `gocbRef`, `goID`, `batch_index`,
every SV value, `frameLen`, `APDUSize`, `GooseTimestamp`. So the pair is
**not** two subscribers seeing the same message (the MACs would differ), and
**not** two SV cycles correlated to one GOOSE message (the SV values would
differ). It is the same message, written twice, with the attack label applied
to one copy and `normal` to the other.

## 3b. Root cause: ERENO models a *dropping* attack as an *emitting* one

`BalancedSamambaiaScenario` puts both the legitimate IED and the attack IED on
the station bus and writes **both** streams into the dataset
(`attacks.legitimate=true`, `attacks.orientedGrayhole=true` in
`attacks.properties`). And `OrientedGrayHoleIED` is fed the legitimate
stream verbatim:

    messageCreator = new OrientedGrayHoleCreator(legitimateIED.copyMessages());
    messageCreator.generate(this, discardRate);

It copies the legitimate messages, drops some, and re-emits the survivors
under the attack label. So the dataset is the legitimate stream (labelled
`normal`) unioned with unmodified copies of the same messages (labelled
`SAG.*`/`FRG`). The "duplicate" is not a writer slip - it is one message
observed twice, once per IED.

**This is a design mismatch, not a coding bug, and that is what makes it
serious.** ERENO's model is "an attacker IED emits labelled messages", which
works for every other attack in the registry - replay, injection, masquerade,
high-StNum, flooding - because those attackers emit something *different* from
the legitimate traffic, so the duplicate carries real signal.

A grayhole emits nothing. It withholds. The information about it is in the
**absence** of messages, and an emitting-attacker model cannot label an
absence. What it labels instead is a forwarded copy that is byte-identical to
benign traffic - which is exactly what section 1 measures.

Setting `attacks.legitimate=false` would remove the twins but not fix this: it
would leave a dataset whose `normal` class is gone from that publisher and
whose attack class is a stream of ordinary messages. The label would still not
be a function of the features.

## 4. It was introduced by the regeneration

The legacy dataset behind the submitted paper does not have it:

| Dataset | rows | distinct message keys | attack rows sharing a key with `normal` |
|---|---:|---:|---:|
| `data/CSV files/gray-GOOSE.csv` (submitted paper) | 1,006,989 | 1,006,981 | 15 (0.004%) |
| `data/runs/` (regenerated, 265 runs) | 23,226,530 | ~half | **100%** |

So the submitted paper's results were not affected by this, and some
configuration exists under which ERENO does not produce it. **Which one is
not determined here**: the legacy run parameters are unrecoverable
(`data_card.md` - legacy ERENO called `new Random(System.nanoTime())` and
recorded neither seed nor scenario), so whether the difference is the
`attacks.legitimate` flag, a different scenario class, or one run per attack
variant instead of all four together cannot be read off the data. That
question should be answered before regenerating, because the answer is the
cheapest available fix if one of those settings is sufficient.

## 5. What it invalidates

**Directly:**

- Every per-class attack precision, recall, F1 and average precision in
  `validation_protocol.md` (cards D.3, E, and the threshold axis) and in
  `ablations_baselines.md` §9/§11.
- The card-E conclusion that balancing is what moves attack detection, and the
  D.3 conclusion that model capacity is the axis that matters. Both compared
  models on a target that is partly unlearnable by construction.
- The card-C benign-confound reading. The same defect affects
  `benign_degradation`: in the loss-type mechanisms ~90% of degraded rows have
  a `normal` twin.

**Not invalidated:**

- The grouped-splitting protocol, the leakage audit and the run-level
  bootstrap. Those are about *how* the evaluation is run, and they are
  unaffected by what the labels say.
- The threshold-axis tooling (`grouped_pr_curves.py`) and the finding that
  argmax reporting hides the operating point. That argument is about
  reporting, not about this dataset.
- The loader, the posterior persistence and the integrity audits.

**Sharpened rather than invalidated:** the observation that the attack signal
lives entirely in the delta columns. That is now explained - the content
features *cannot* carry it, because every attack row's content also occurs
under `normal`. What looked like "the model keys on sequence gaps" may be
partly "the model keys on whether this row is the second copy", which is a
property of the writer.

## 6. What has to happen before any more model runs

1. **Decide what a grayhole row *is*** (see §3b and §7). Neither available
   switch fixes it: keeping both streams gives identical rows under two
   labels, and dropping the legitimate stream gives an attack class made of
   ordinary messages. The generator has to be changed to express a withheld
   message, or the label has to move off the message.
2. **Regenerate the 265-run matrix** once the generator is fixed. The run
   matrix, the seeds and the grouped protocol are all reusable; only the CSVs
   change.
3. **Add `check_label_duplication.py` to the chain as a gate**, the way
   `check_no_leakage.py` gates training. Nothing in the existing chain checks
   this: `check_no_leakage.py` owns train/test group overlap,
   `merge_runs.py` owns cross-run payload sharing, and
   `check_prediction_integrity.py` audits predictions after the fact. All
   three pass on this dataset.
4. **Re-run cards D and E** on the regenerated pool. The tooling is unchanged,
   so this is compute, not rework.

## 7. The question this forces, and it is the paper's question

§3b says the generator cannot express a dropping attack at message level.
That is not a tooling problem to route around - it is the same problem the
detector has. If a grayhole withholds messages, then:

- the dropped message is not in the capture, so it cannot carry a label;
- the messages that *are* captured are ordinary traffic, so labelling them
  "attack" attaches a label their content cannot support;
- the only observable is a property of the **sequence** - a gap, a stretched
  interval - which is a property of a window, not of a row.

So the experimental unit is probably wrong. A defensible redesign labels a
**window** or a **trace segment** as attacked/not, and the model classifies
windows. That also changes what SHAP explains (card F) and what the
rule-based baseline compares against (card D.2), and it makes the delta
features first-class rather than a leaky proxy.

This should be decided **before** regenerating, because a regeneration is
expensive and there is only reason to do it once.

## 8. Reproducing this

    python experiments/revision_2026/check_label_duplication.py \
      --runs-dir data/runs \
      --out experiments/revision_2026/label_duplication.md \
      --report experiments/revision_2026/label_duplication.json

    python experiments/revision_2026/check_label_duplication.py \
      --dataset data/runs/gray-GOOSE-runs-prepared.parquet

Exit code 1 on any content twin, so it can gate a regeneration.
