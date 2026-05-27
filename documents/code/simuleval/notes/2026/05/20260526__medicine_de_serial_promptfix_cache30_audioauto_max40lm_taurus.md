## Hypothesis

Serial medicine hardraw En-De with the prompt-history fix should provide a clean
comparison against the batch medicine results under the same cache budget.

## Background / Motivation

The batch medicine de curve has already been produced for cap16-denoise RASST.
The user wants the serial readout as a direct comparison, using the same
paper-facing runtime policy as the repaired tagged-ACL serial run:
`max_cache_chunks=30`, `keep_cache_chunks=30`, `VLLM_LIMIT_AUDIO=auto`,
`empty_term_map_policy=omit`, and `max_new_tokens=40*lm`.

## What changed vs baseline

- Run medicine hardraw En-De only.
- Use the cap16-denoise tagged-term German SLM and HN1024 retriever.
- Use the fixed serial agent in `agents/infinisst_omni_vllm_maxsim_rag.py`.
- Validate every runtime prompt has exactly one system prompt and no more than
  30 audio chunks.

## Expected metrics

The run should produce one row per latency multiplier with five medicine talks
in both `instances.log` and `instances.strip_term.log`. Metrics should be
comparable to batch outputs, with any difference attributable to serial vs batch
execution rather than prompt-history duplication.

## Verdict

Pending.
