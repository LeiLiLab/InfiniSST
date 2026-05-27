## Hypothesis

The `lh1b88kw` HN1024 checkpoint should remain the recall-oriented reference
line, especially at the previously selected downstream tau `0.73`.

## Background / Motivation

Prior readouts established `lh1b88kw` at tau `0.73` as the main retriever result
for speech-LLM background work.  This eval reruns the same broad dev / ACL /
tagged-ACL / strict-medicine tau-delta surface used for the no-HN checkpoint so
the comparison is not mixed across different eval jobs or data paths.

## What changed vs baseline

- Checkpoint: `lh1b88kw` HN1024 best-secondary checkpoint.
- Dev surface: raw/base, gs10k, and gs100k recall plus tau-filtered
  precision/recall for tau `0.70..0.90`.
- Comparison rule: do not reselect `lh1b88kw` tau from this held-out-facing
  comparison; report tau `0.73` as the fixed main-result setting.
- Held-out readouts: ACL6060, tagged ACL6060, and strict MFA-only medicine.

## Expected metrics

The primary comparison row is `lh1b88kw` at tau `0.73`, with no-HN compared at
its dev-selected tau under the predeclared `0.5pp` maximum dev recall-drop rule.

## Verdict

Finished as W&B run `v4vl6zxr`, but this run is now marked legacy diagnostic
only.  It used the old single-glossary eval path (`*_eval_wiki_glossary` plus
`*_eval_glossary_sizes`) rather than the upgraded split between a fixed strict
metrics glossary and a variable retriever glossary.

The old path rebuilds the positive set after expanding the retrieval bank, so
`gs1k/gs10k` recall is not measured against a fixed strict raw denominator.
This can make an expanded-bank metric such as ACL gs10k recall exceed ACL raw
recall, which is a denominator/protocol artifact rather than a model result.

Do not use the ACL/tagged-ACL/medicine numbers from this run for paper
comparison.  The `lh1b88kw` checkpoint and tau sweep are still useful as a
diagnostic artifact, but the formal HN ablation needs to be rerun with the
metrics-glossary / retriever-glossary split.
