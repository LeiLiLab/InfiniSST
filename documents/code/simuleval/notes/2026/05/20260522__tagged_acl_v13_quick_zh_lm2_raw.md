## Hypothesis

V13 trains the speech LLM with inference-aligned retriever timeline term maps on `lm=1..6` chunks. This quick readout checks whether that data construction improves Chinese tagged-ACL adoption on the main `lm=2/raw` setting.

## Background / Motivation

Previous retriever-SFT variants underperformed the no term-map-SFT baseline. V13 removes long `lm>6` training chunks and uses the current retrieval policy more directly: low-latency chunks wait until enough timeline context is available, while chunks at least `2.88s` are retrieved directly.

## What changed vs baseline

- Speech LLM checkpoint: V13 `lm=1..6` retriever-timeline SFT, LoRA rank 8 alpha 32.
- Evaluation setting: tagged ACL, Chinese, latency multiplier 2, raw strict glossary.
- Retriever: `lh1b88kw` best ACL6060 checkpoint.
- Retrieval: timeline mode, lookback `1.92s`, tau `0.73`, top-k `10`.
- Term-map format: plain.
- Term metrics: fixed raw tagged ACL denominator; FCR policy `term_map_source_ref_negative_sentence`.

## Expected metrics

Primary metrics are BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL. The immediate comparison point is the existing no term-map-SFT `zh lm=2 raw` readout.

## Verdict

Completed on 2026-05-22. V13 did not improve the `zh lm=2 raw` setting: BLEU 45.26, TERM_ACC 78.09%, REAL_TERM_ADOPT 78.07%, TERM_FCR 8.31%, StreamLAAL 1718.38. The output log shows repeated long-prefix carryover, so this variant is not a good candidate for scaling.
