## Hypothesis

The previous serial main-result `de` readout regressed because the serial agent
duplicated the system prompt while trimming vLLM prompt history. After the agent
fix, each `de` latency setting should run with exactly one system prompt per
vLLM request.

## Background / Motivation

The old batch `de/lm=3` RASST result used one system prompt in every request.
The new serial run showed two system prompts at segment 0 and accumulated up to
22 system prompts later in a talk. Source order and `TERM_TOTAL=935` were stable,
so this rerun isolates the serial prompt-history fix before rerunning broader
main-result grids.

## What changed vs baseline

- Patched `agents/infinisst_omni_vllm_maxsim_rag.py` to trim only the message
  body after the initial system message.
- Rerun tagged ACL raw `de` latency settings with the prompt-history fix.
- Keep the intended serial paper settings:
  - `max_cache_chunks=30`
  - `keep_cache_chunks=30`
  - `VLLM_LIMIT_AUDIO=auto`, resolved by the agent to `max_cache_chunks=30`
  - `empty_term_map_policy=omit`
  - `system_prompt_style=given_chunks`
  - `max_new_tokens=40*lm=120`

## Expected metrics

BLEU should recover relative to the broken serial row and TERM_ACC should stay
near the cap16-denoise tagged ACL curve. The first validation gate is runtime
prompt integrity: every logged `llm_input` must contain exactly one system
prompt.

## Verdict

Pending.
