## Hypothesis
JA medicine hardraw serial SimulEval should be compared against the concurrently running same-LM batch readout for lm=1 and lm=2 under the same promptfix and VLLM audio-limit setting.

## Background / Motivation
The current JA medicine batch readout uses the cap16-denoise tagged-term SLM with `new_promptfix_vllmaudio128`. We need serial lm=1/lm=2 rows to determine whether the batch path changes BLEU, latency, or term accuracy.

## What changed vs baseline
This launcher runs one LM per process in serial SimulEval on Aries. It uses the Aries-visible local JA cap16-denoise HF checkpoint, HN1024 retrieval at tau 0.78, `empty_term_map_policy=omit`, `rag_prompt_policy=given_chunks`, `VLLM_LIMIT_AUDIO=128`, `max_cache_chunks=30`, `keep_cache_chunks=30`, and `max_new_tokens=40*lm`.

## Expected metrics
Each lm should produce one `eval_results.tsv`, plus `instances.log` and `instances.strip_term.log` with five rows over the medicine hardraw five-talk set.

## Verdict
Pending.
