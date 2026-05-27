# Tagged ACL V16 Fixed-Denominator Raw/10k and Extracted110 Eval

## Hypothesis

V16 LLM-variant and V16 no-GT-zero should be compared under fixed metric denominators while changing only the runtime glossary size.

## Background / Motivation

The user requested `v16_llmvariant` and `v16_no_gt_zero` evals with strict fixed raw glossary denominator for runtime `raw` and `gs10k`, plus the paper110 extracted-glossary reference with a matched extracted-gs10k noise control.

## What changed vs baseline

- Raw panel: runtime glossary `raw` and `gs10k`; metric denominator fixed to `acl6060_tagged_gt_raw_min_norm2.json`.
- Extracted110 panel: runtime glossary `extracted` and `extracted_gs10k`; metric denominator fixed to `extracted_glossary__2022.acl-long.110.json`.
- Language/latency: `zh`, `lm=2`.
- Retriever: `lh1b88kw`, top-10, tau=0.73, timeline lookback=1.92s.

## Expected metrics

Report BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL for each runtime glossary.  For raw-vs-gs10k comparisons, TERM-related denominators must remain fixed.

## Verdict

Completed.  All eight W&B eval runs finished with `status:success`.
Under the fixed raw tagged denominator, `v16_no_gt_zero` improves over
`v16_llmvariant` on `zh lm=2`: raw TERM_ACC/REAL_TERM_ADOPT improve from
83.37/83.27 to 87.75/89.10, and gs10k improves from 84.04/82.78 to
87.98/89.28.  This supports the current interpretation that the positive effect
comes from stacking no-GT chunk zeroing on top of the LLM-variant term
translation augmentation, while the extracted110 panel is only a small
reference readout.
