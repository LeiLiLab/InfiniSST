## Hypothesis

The `lh1b88kw` HN1024 checkpoint should be re-evaluated with the same fixed
strict raw/base metrics denominator used for the no-HN rerun, so the HN
ablation is not affected by changing positive universes across retriever bank
sizes.

## Background / Motivation

The prior `lh1b88kw` comparison run `v4vl6zxr` used the legacy
dynamic-denominator eval path and is marked diagnostic only.  This rerun keeps
the metrics denominator fixed to the raw/base strict glossary while allowing
the retriever candidate bank to change for gs1k / gs10k / gs100k.

## What changed vs baseline

- Checkpoint: `lh1b88kw` HN1024 best-secondary checkpoint.
- Eval protocol: `eval_metric_denominator=fixed_raw`.
- Retriever banks: dev base / gs10k / gs100k; ACL/tagged-ACL/medicine raw /
  gs1k / gs10k.
- Held-out readouts: ACL6060, tagged ACL6060, and strict medicine remain
  readout-only.

## Expected metrics

Expanded-bank recall should no longer exceed raw recall due to denominator
changes.  Compare HN1024 against no-HN by matching dev max recall drop, not by
raw tau.

## Verdict

Finished as W&B run `ry8osg4u` with `status:success`.

The fixed-denominator sanity check passes.  HN1024 improves precision at more
aggressive thresholds, but at matched dev max-drop it loses held-out recall
against no-HN on ACL6060, tagged ACL, and medicine.  See
`documents/code/train/term_train/reports/20260522_nohn_vs_hn1024_fixeddenom_eval_report.md`.
