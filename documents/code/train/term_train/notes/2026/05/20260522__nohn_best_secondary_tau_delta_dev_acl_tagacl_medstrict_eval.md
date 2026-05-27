## Hypothesis

The no-HN best-secondary checkpoint may recover enough dev recall without hard
negatives to challenge the current `lh1b88kw` main result, but its inference
threshold must be chosen from dev only before ACL/tagged-ACL/medicine readouts.

## Background / Motivation

The no-HN training run `40fgbr2y` was manually stopped after convergence and
uses the checkpoint selected by `eval_acl6060/recall@10` at step 1600.  The
current main result uses the `lh1b88kw` HN1024 checkpoint and downstream tau
`0.73`.  This eval reruns the broader tau-delta readout on the no-HN
checkpoint with dev base / gs10k / gs100k, paper ACL, tagged ACL, and strict
MFA-only medicine.

## What changed vs baseline

- Checkpoint: no-HN `40fgbr2y` canonical best-secondary checkpoint.
- Dev calibration: evaluate raw/base, gs10k, and gs100k recall plus tau-filtered
  precision/recall for tau `0.70..0.90`.
- Frozen selection rule: choose the largest no-HN tau whose maximum dev recall
  drop across base / gs10k / gs100k is at most `0.5pp`.
- Held-out readouts: ACL6060, tagged ACL6060, and strict MFA-only medicine are
  reported after the dev rule and must not choose tau.

## Expected metrics

The key comparison is no-HN at its dev-selected tau against `lh1b88kw` at
tau `0.73`, with raw recall kept visible for base / gs10k / gs100k dev and
precision/recall reported for ACL, tagged ACL, and strict medicine.

## Verdict

Finished as W&B run `vj0z7xdv`, but this run is now marked legacy diagnostic
only.  It used the old single-glossary eval path (`*_eval_wiki_glossary` plus
`*_eval_glossary_sizes`) rather than the upgraded split between a fixed strict
metrics glossary and a variable retriever glossary.

The old path rebuilds the positive set after expanding the retrieval bank, so
`gs1k/gs10k` recall is not measured against a fixed strict raw denominator.
This can make an expanded-bank metric such as ACL gs10k recall exceed ACL raw
recall, which is a denominator/protocol artifact rather than a model result.

Do not use the ACL/tagged-ACL/medicine numbers from this run for paper
comparison.  The no-HN checkpoint and tau sweep are still useful as a
diagnostic artifact, but the formal HN ablation needs to be rerun with the
metrics-glossary / retriever-glossary split.
