# Tagged ACL Same-LM Batch V1 ja lm2 raw

## Hypothesis

Batching the five tagged-ACL talks for fixed `lm=2` should give a fast readout for the clean ja New V9 Speech LLM while preserving the same HN1024 tau=0.78 retrieval setup.

## Background / Motivation

The zh same-lm batch prototype is usable for accelerated evaluation when each run keeps one latency multiplier.  This run applies the same path to En-Ja tagged ACL raw glossary with the clean MFA+OpenAI New V9 Speech LLM.

## What changed vs baseline

- Runs only `ja`, `lm=2`, raw tagged ACL glossary.
- Uses the clean New V9 MFA+OpenAI ja Speech LLM export when available.
- Uses HN1024 retriever at tau `0.78`, lookback `1.92s`, top-k `10`.
- Uses fixed `max_new_tokens=40`, same decoding settings, and assistant `<term>` stripping.
- Uses batched same-lm vLLM scheduling with batched timeline retrieval enabled.

## Expected metrics

Metrics should be plausible for ja raw lm2 and suitable as a fast sanity check.  If a serial clean-ja lm2 run is later available, compare against it before treating batch output as exact serial replacement.

## Verdict

Completed.  The first same-lm batch pass wrote a valid `eval_results.tsv` for
ja lm2 raw tagged ACL.  A duplicate second pass started from the same launcher
after the first output was written; it was stopped before completion to avoid
wasting GPU time.

Key metrics from the completed output:

- BLEU: 27.26
- TERM_ACC: 83.51%
- REAL_TERM_ADOPT: 87.45%
- TERM_FCR: 19.48%
- StreamLAAL: 1924.98
