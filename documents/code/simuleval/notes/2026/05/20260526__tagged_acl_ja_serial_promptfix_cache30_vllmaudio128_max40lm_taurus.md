## Hypothesis

Serial tagged-ACL En-Ja with the fixed prompt-history agent and
`VLLM_LIMIT_AUDIO=128` should provide the direct comparison requested against the
earlier main-result JA rows.

## Background / Motivation

The current main-result JA tagged-ACL rows use the NewV9 assistant-term-tag-delay
Speech LLM. After debugging the German serial path, the user requested rerunning
the four JA latency multipliers on Taurus with the same serial settings.

## What changed vs baseline

- Use the fixed `agents/infinisst_omni_vllm_maxsim_rag.py` prompt-history logic.
- Run tagged ACL raw En-Ja for `lm=1,2,3,4`.
- Keep the main-result runtime policy:
  - `max_cache_chunks=30`
  - `keep_cache_chunks=30`
  - `VLLM_LIMIT_AUDIO=128`
  - `empty_term_map_policy=omit`
  - `system_prompt_style=given_chunks`
  - `max_new_tokens=40*lm`

## Expected metrics

Each latency multiplier should produce `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log` with five ACL talks. Runtime prompt validation should
confirm one system prompt per `llm_input`.

## Verdict

Pending.
