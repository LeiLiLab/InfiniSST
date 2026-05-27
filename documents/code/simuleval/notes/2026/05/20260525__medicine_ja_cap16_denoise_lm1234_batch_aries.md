## Hypothesis
JA cap16-denoise should be evaluated on medicine hardraw for lm=1,2,3,4 after the DE tagged ACL lm=2/3 readout completes.

## Background / Motivation
The JA cap16-denoise model is the short-tag term-wrapper variant trained from the cap16-denoise data branch. The medicine readout uses the manually checked hardraw denominator and same-LM batch-vLLM evaluation.

## What changed vs baseline
This run uses the Aries local HF cache for JA cap16-denoise, HN1024 retrieval at tau 0.78, `empty_term_map_policy=omit`, `strip_output_tags=term_t`, `max_cache_chunks=30`, `keep_cache_chunks=30`, and `max_new_tokens=20*lm`.

## Expected metrics
The run should produce medicine hardraw `eval_results.tsv`, `instances.log`, and `instances.strip_term.log` for lm=1,2,3,4. Each log should contain five rows.

## Verdict
Pending.
