## Hypothesis

The no-HN best-secondary checkpoint may have a raw/base-included `<0.5pp`
dev max-drop operating point below tau `0.65`.  The previous `0.65..0.90`
dev-1M sweep showed no valid raw-included `<0.5pp` tau for no-HN because
tau `0.65` dropped raw/base recall by about `0.74pp`.

## Background / Motivation

The fixed-denominator no-HN vs HN1024 report now separates two calibration
surfaces: raw/base-included max-drop and expanded-bank-only max-drop.  The
expanded-bank-only surface found no-HN tau `0.65` for `<0.5pp`, but the
raw/base-included surface still needed lower tau values.

This eval explores tau `0.50..0.70` at stride `0.01` on the same dev-only
dev-1M setup as run `evcgcdlu`.

## What changed vs baseline

- Checkpoint: no-HN `40fgbr2y` best-secondary checkpoint.
- Tau grid: `0.50..0.70`, stride `0.01`.
- Dev retriever banks: gs10k, gs100k, and gs1M from the P31 untrained 1M
  glossary source.
- Metrics denominator: fixed raw/strict dev positives; retriever glossary size
  changes only the candidate bank.
- Held-out readouts: disabled.

## Expected metrics

The key output is the highest tau satisfying raw/base-included max dev recall
drop `<0.5pp`, with raw tau `0.0` recall, filtered recall, micro precision,
and kept counts visible for raw, gs10k, gs100k, and gs1M.

## Verdict

Completed as W&B run `e8t8zdtd`.

The lower tau sweep found a raw/base-included strict `<0.5pp` dev operating
point for no-HN:

- tau `0.61`, max drop `0.4537pp`.
- raw/base: recall `98.7982`, micro precision `12.7427`, kept `7.77`.
- gs10k: recall `98.7345`, micro precision `11.3323`, kept `8.73`.
- gs100k: recall `98.5037`, micro precision `10.1788`, kept `9.68`.
- gs1M: recall `97.9545`, micro precision `9.8216`, kept `9.98`.

For `<1.0pp`, no-HN selects tau `0.69` in this lower sweep.  For `<1.5pp`,
the previous high sweep `evcgcdlu` still provides the higher tau `0.73`.
Conclusion: no-HN does have a strict `<0.5pp` point once tau is extended below
`0.65`, but HN1024 reaches the same strict budget at a much higher tau
(`0.72`), so HN's clearest advantage is thresholdability/calibration rather
than raw recall.
