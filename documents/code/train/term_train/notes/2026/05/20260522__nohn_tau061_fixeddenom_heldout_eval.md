## Hypothesis

After dev-only calibration, no-HN tau `0.61` is the strict raw-included
`<0.5pp` operating point.  A held-out readout at this frozen tau should be
compared against HN1024 tau `0.72`, the matched strict `<0.5pp` HN point.

## Background / Motivation

The previous fixed-denominator held-out no-HN run used tau `0.70..0.90`, so it
did not contain the newly selected no-HN tau `0.61`.  HN1024 tau `0.72` is
already present in run `ry8osg4u`.

This run fills only the missing no-HN held-out readout.  Tau selection remains
dev-only and was already frozen before reading held-out metrics.

## What changed vs baseline

- Checkpoint: no-HN `40fgbr2y` best-secondary checkpoint.
- Tau grid: single tau `0.61`.
- Readouts: dev, ACL6060 paper glossary, tagged ACL6060, and strict medicine.
- Metrics denominator: fixed raw/strict term universe; retriever glossary size
  changes only the candidate bank.

## Expected metrics

The output should provide no-HN `tau=0.61` recall, micro precision, and kept
counts for held-out ACL/tagged/medicine, especially gs10k, so the report can
compare no-HN `0.61` against HN1024 `0.72`.

## Verdict

Completed as W&B run `zji769ve`.

Strict fixed-denominator gs10k held-out readout at no-HN tau `0.61`:

| readout | R | P_micro |
|---|---:|---:|
| ACL6060 | 94.63 | 9.52 |
| tagged ACL6060 | 98.07 | 9.91 |
| medicine strict | 94.10 | 10.34 |

Against HN1024 tau `0.72` from run `ry8osg4u`, the deltas
`HN1024 - no-HN` are:

| readout | delta R | delta P_micro |
|---|---:|---:|
| ACL6060 gs10k | -1.52 | -0.06 |
| tagged ACL6060 gs10k | -0.63 | +0.05 |
| medicine gs10k | -1.41 | +0.56 |

At the matched strict `<0.5 pp` dev-drop operating point, HN1024 remains a
thresholding/precision tradeoff rather than a held-out recall win.
