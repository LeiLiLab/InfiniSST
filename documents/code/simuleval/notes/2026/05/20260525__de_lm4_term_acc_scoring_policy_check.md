# En-De lm4 TERM_ACC Scoring Policy Check

## Hypothesis

The serial and batch En-De lm4 no-RAG evals use the same `TERM_ACC` denominator:
a glossary item is counted only when the source sentence contains the source
term and the target reference sentence contains the target translation.

## Background / Motivation

The batch and serial lm4 no-RAG results have different `TERM_ACC`, so the
scoring denominator and scorer path need to be checked before attributing the
difference only to generation behavior.

## What changed vs baseline

No generation or scoring outputs were changed. This analysis inspected
manifests, eval TSVs, eval logs, launcher command lines, `offline_streamlaal_eval.py`,
the called FBK `stream_laal_term.py`, and source/ref/audio/glossary artifacts.

## Expected metrics

No new score is introduced. The expected denominator from independent
source/reference/glossary matching is `TERM_TOTAL=935`.

## Verdict

`TERM_ACC` is consistent across the two lm4 TSVs. Both use the same raw tagged
glossary and equivalent source/reference/audio metadata. The scorer increments
the denominator only when the source contains the English term and the target
reference contains the German translation. The different `TERM_ACC` is caused by
different predictions, not by a denominator-policy mismatch. The batch run has
missing optional `TERM_ADOPTION` fields because the repo-local adoption script
path was wrong on Aries, but that does not affect `TERM_ACC`.

Report:
`documents/code/simuleval/reports/20260525_de_lm4_term_acc_scoring_policy_check.md`
