## Hypothesis

The earlier En-De lm4 InfiniSST/no-RAG batch result may not be directly comparable to the serial verified baseline because batch used a different effective decoding/cache setup. This rerun aligns the batch configuration to the serial lm4 rerun.

## Background / Motivation

The verified serial baseline for tagged ACL En-De lm4 is BLEU 33.3008 from `20260524T160830__simuleval__tagged_acl_origin_norag_de_lm4_raw_rerun`. That serial launcher used fixed `MAX_NEW_TOKENS=40` and cache defaults `80s/60s`, which are equivalent to about `20/15` chunks at lm4. Some later batch runs used `40*lm` decoding or larger audio prompt limits, so the batch/serial difference should be rechecked under aligned settings.

## What changed vs baseline

- Runs batch vLLM only, not serial SimulEval.
- Keeps the origin German Speech LLM and disables RAG.
- Uses `NORAG_PROMPT_POLICY=serial_compat`.
- Uses fixed `max_new_tokens=40`, not `40*lm`.
- Uses `max_cache_seconds=80`, `keep_cache_seconds=60`, `min_cache_chunks=1`.
- Uses `vllm_limit_audio=20`, matching the lm4 max-cache chunk budget.

## Expected metrics

If parameter mismatch explains the gap, this batch result should move toward the serial lm4 BLEU 33.3008. If it remains much lower, the remaining difference is likely from batch execution semantics rather than cache/decode settings alone.

## Verdict

Pending.
