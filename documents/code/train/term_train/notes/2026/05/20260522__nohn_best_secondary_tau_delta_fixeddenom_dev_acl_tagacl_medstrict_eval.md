## Hypothesis

The no-HN best-secondary checkpoint should be re-evaluated with a fixed strict
raw/base metrics denominator, because the old single-glossary eval path allowed
expanded retriever banks to change the positive universe.

## Background / Motivation

The prior no-HN comparison run `vj0z7xdv` used the legacy dynamic-denominator
eval path and is marked diagnostic only.  This rerun keeps the metrics
denominator fixed to the raw/base strict glossary while allowing the retriever
candidate bank to change for gs1k / gs10k / gs100k.

## What changed vs baseline

- Checkpoint: no-HN `40fgbr2y` best-secondary checkpoint at step 1600.
- Eval protocol: `eval_metric_denominator=fixed_raw`.
- Retriever banks: dev base / gs10k / gs100k; ACL/tagged-ACL/medicine raw /
  gs1k / gs10k.
- Held-out readouts: ACL6060, tagged ACL6060, and strict medicine remain
  readout-only.

## Expected metrics

Expanded-bank recall should no longer exceed raw recall due to denominator
changes.  Tau selection should be based on dev max recall drop under the fixed
raw/base denominator.

## Verdict

Finished as W&B run `9esujv2w` with `status:success`.

The fixed-denominator sanity check passes: expanded retriever banks no longer
increase recall over the raw/base bank by changing the metric denominator.  In
the corrected comparison, no-HN is not weaker on unfiltered recall and remains
the safer recall-first choice.  See
`documents/code/train/term_train/reports/20260522_nohn_vs_hn1024_fixeddenom_eval_report.md`.
