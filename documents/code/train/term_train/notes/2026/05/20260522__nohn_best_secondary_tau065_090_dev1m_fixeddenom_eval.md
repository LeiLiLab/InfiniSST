## Hypothesis

The no-HN best-secondary checkpoint may have a recall-retaining operating
point below tau 0.70 once the sweep grid is expanded to tau `0.65..0.90` at
0.01 stride.

## Background / Motivation

The prior fixed-denominator no-HN vs HN1024 comparison swept tau `0.70..0.90`.
That grid was too narrow for no-HN when choosing tau by maximum dev recall drop,
so the dev calibration surface needs to be extended downward before any
held-out readout is interpreted.

This run also probes whether the dev calibration rule changes when the retriever
bank is expanded from 100k to 1M candidates.  ACL, tagged ACL, and medicine are
disabled in this run because the selection rule must be dev-only.

## What changed vs baseline

- Checkpoint: no-HN `40fgbr2y` best-secondary checkpoint.
- Tau grid: `0.65..0.90`, stride `0.01`.
- Dev retriever banks: gs10k, gs100k, and gs1M from the P31 untrained 1M
  glossary source.
- Metrics denominator: fixed raw/strict dev positives; retriever glossary size
  changes only the candidate bank.
- Held-out readouts: disabled.

## Expected metrics

The main readout is the no-HN tau chosen by max dev recall drop thresholds
`<0.5pp`, `<1.0pp`, and `<1.5pp`, with raw tau `0.0` recall kept visible for
dev base, gs10k, gs100k, and gs1M.

## Verdict

Finished as W&B run `evcgcdlu`.

The expanded tau grid and dev-1M readout completed successfully with fixed raw
metrics denominator.  The key dev unfiltered recalls were base `0.9924`, gs10k
`0.9898`, gs100k `0.9858`, and gs1M `0.9799`.

With raw/base included in the max-drop rule, the requested `0.65..0.90` grid
still has no no-HN operating point below `<0.5pp`: tau `0.65` drops base recall
by about `0.74pp`.  If the selection rule is applied only to the expanded
retriever banks, then tau `0.65` is valid for `<0.5pp` because the gs10k /
gs100k / gs1M drops are about `0.49pp`, `0.19pp`, and `0.06pp`.  Report both
surfaces explicitly so the raw/base strict-denominator constraint is not hidden.
