## Hypothesis

The earlier En-De tagged-ACL RASST result may have appeared to recover BLEU because
the runtime protocol differed from the current batch setting, not because the
retriever output changed. A small lm=4 compatibility matrix can isolate whether
old tau/cache/max-token/empty-map behavior explains the gap.

## Background / Motivation

The old LLM-generated term-map SFT / TM-SFT Speech LLM with HN1024 reported
strong de/lm=4 BLEU while keeping high term accuracy. Current reruns with the
same model and HN1024 at tau=0.78 plus omit-empty-map remained below the verified
no-RAG BLEU gate. MFA analysis indicates term-map noise exposure is a likely
SLM-side issue, but before changing data again we need to reproduce the old
runtime protocol in the current batch evaluator.

## What changed vs baseline

This is a standalone batch SimulEval readout for de/lm=4 only. It uses the old
TM-SFT / LLM-generated term-map Speech LLM, HN1024 retriever, tagged ACL raw
glossary, five talks, and batch vLLM. Four protocol settings are compared:

- tau=0.73, max_new_tokens=40, old 80/60s cache with 20/15 chunk limits,
  empty term maps rendered as `term_map:\nNONE`.
- tau=0.73, max_new_tokens=40, old 80/60s cache with 20/15 chunk limits,
  empty term maps omitted.
- tau=0.73, max_new_tokens=80, short 40/20s cache with 8/4 chunk limits,
  empty term maps omitted.
- tau=0.78, max_new_tokens=80, old 80/60s cache with 20/15 chunk limits,
  empty term maps omitted.

The eval remains a held-out readout; these settings are diagnostic, not a new
calibration rule.

## Expected metrics

If the old win was mostly protocol-driven, one of the tau=0.73 or old-cache
settings should recover BLEU near the historical de/lm=4 value while preserving
TERM_ACC around 0.84. If all settings remain near the current tau=0.78 omit
rerun, then the old apparent gain was probably tied to the old serial runner,
old comparison baseline, or a stale artifact rather than an easy runtime knob.

## Verdict

Completed on Aries. The old protocol knobs do not recover BLEU above the
verified no-RAG lm=4 gate. The best setting is `tau=0.73`, short `40/20s`
cache, `8/4` chunk limits, `max_new_tokens=80`, and omitted empty maps:
BLEU `32.7702`, TERM_ACC `0.8492` (`794/935`). This improves over the
tau=0.78 old-cache omit setting (`32.5332`) but remains below the verified
no-RAG gate (`33.3008`) and below the current cap16 selected short-cache result
(`33.4820`, TERM_ACC `0.8674`).

All four rows wrote one `eval_results.tsv`, five `instances.log` rows, and five
`instances.strip_term.log` rows. Optional `REAL_TERM_ADOPT` and `TERM_FCR` are
`N/A` because the optional post-hoc adoption script path was not found inside
the Aries child process; headline `TERM_ACC` is unaffected.
