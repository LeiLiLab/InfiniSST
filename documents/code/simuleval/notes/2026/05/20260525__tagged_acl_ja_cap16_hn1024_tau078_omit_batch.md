# Tagged ACL ja cap16 HN1024 tau0.78 omit batch

## Hypothesis

The Japanese retriever-cap16 exact-boundary speech LLM may reduce term-map noise
relative to prior JA RASST variants while keeping HN1024 retrieval active.

## Background / Motivation

Run tagged ACL JA held-out readout for `lm=1,2,3,4` using the exported HF SLM
from `speech_llm_ja_retriever_cap16_exactboundary_r32a32_ep4_aries8`.
The runtime policy should omit the empty `term_map:NONE` user block while still
passing non-empty HN1024 retrieved term maps.

## What changed vs baseline

- Speech LLM: JA retriever-cap16 exact-boundary r32/a32 ep4 Aries-8 export.
- Retriever: HN1024 checkpoint with tau=0.78 and 1.92s lookback.
- Runtime: batched vLLM same-LM eval with `empty_term_map_policy=omit`.
- Dataset: tagged ACL raw glossary held-out readout, five ACL talks per LM.

## Expected metrics

Compare BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, and TERM_FCR against the
prior JA tagged ACL HN1024 tau0.78 readouts. This is an ACL readout only; do
not use it to choose model or tau.

## Verdict

Partial completion on 2026-05-25. `lm=3` and `lm=4` completed with verified
`eval_results.tsv`, `instances.log`, and `instances.strip_term.log` artifacts.
`lm=1` and `lm=2` failed before eval artifact creation during vLLM KV-cache
initialization with `ShmRingBuffer` shared-memory errors, so no full `lm=1..4`
average is available from this batch.

Authoritative metric paths and W&B run ids are recorded in the event manifest;
this notes file is only the intent/verdict record.
