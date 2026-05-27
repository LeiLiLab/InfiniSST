# Tagged ACL quick eval for V15, V16, and New V4

## Hypothesis

The quick eval should identify whether V15 marker augmentation, V16 natural-variant augmentation, or New V4 natural-variant augmentation on the older `new_v3` data improves `zh lm=2 raw` tagged ACL term adoption.

## Background / Motivation

Previous quick checks showed several retriever-SFT variants underperformed the no term-map SFT and LLM-generated term-map SFT baselines.  This eval focuses on one high-signal setting, `zh lm=2 raw`, to decide whether any of the new adoption-focused SFT lines deserves full tagged ACL evaluation.

## What changed vs baseline

- Eval setting:
  `lang=zh`, `lm=2`, `glossary=raw`
- Models:
  V15 marker augmentation, V16 LLM-variant augmentation, and New V4 LLM-variant augmentation on old `new_v3` data.
- Term-map format at inference:
  plain `source=target`; V15 markers and V16/new_v4 variants are training-only.
- Eval launcher waits for each HF export to be complete before starting vLLM.

## Expected metrics

Primary metrics are `TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.  A useful variant should improve `TERM_ACC` or `REAL_ADOPT` without a large BLEU regression.

## Verdict

Pending eval.
