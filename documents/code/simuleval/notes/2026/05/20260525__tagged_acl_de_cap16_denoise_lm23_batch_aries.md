## Hypothesis
DE cap16-denoise should be evaluated for tagged ACL lm=2 and lm=3 under the same batch-vLLM settings used for lm=1/4.

## Background / Motivation
The main DE tagged ACL curve still needs lm=2 and lm=3 for the cap16-denoise SLM. The run uses Aries GPU pairs and keeps `max_cache_chunks=30`, `keep_cache_chunks=30`, `empty_term_map_policy=omit`, and `max_new_tokens=20*lm`.

## What changed vs baseline
This is a same-LM five-talk batch readout using the DE cap16-denoise HF checkpoint with HN1024 retrieval at tau 0.78.

## Expected metrics
The run should produce one `eval_results.tsv`, one `instances.log`, and one `instances.strip_term.log` for each of lm=2 and lm=3, each with five rows.

## Verdict
Pending.
