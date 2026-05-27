# Tagged ACL Quick Eval New V5 no-GT-zero old-new_v3 R32 zh lm2 raw

## Hypothesis

New V5 combines old-new_v3 LLM-variant training with no-GT-zero filtering and
should improve `TERM_ACC` and `REAL_ADOPT` over New V4 without a major BLEU
drop.

## Background / Motivation

The training run `cg5qisu9` failed during TensorBoard logging after saving the
iteration-1000 checkpoint.  This eval uses the recovered HF export on data6.

## What changed vs baseline

- Speech LLM: New V5 no-GT-zero old-new_v3 r32/a64 TP2.
- Eval: tagged ACL `zh`, `lm=2`, raw glossary only.
- Runtime retriever: fixed `lh1b88kw` MaxSim retriever with tau `0.73`.
- Term metrics use the fixed raw tagged ACL denominator.

## Expected metrics

Compare against New V4 r32 old-new_v3+LLM-variant raw and V16 no-GT-zero
LLM-variant raw.  Main metrics are BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, and
StreamLAAL.

## Verdict

Success.  Recovered iteration-1000 checkpoint `cg5qisu9` was exportable and
the tagged ACL quick eval completed as W&B `342oxpmu`.

On `zh lm=2 raw`, New V5 reached BLEU 48.20, TERM_ACC 90.00%, REAL_ADOPT
90.19%, TERM_FCR 7.53%, and StreamLAAL 1663.60.  This is a positive result for
the no-GT-zero + old-new_v3 LLM-variant direction, despite the original training
run being marked failed because data7 filled during TensorBoard logging after
the usable checkpoint was saved.
