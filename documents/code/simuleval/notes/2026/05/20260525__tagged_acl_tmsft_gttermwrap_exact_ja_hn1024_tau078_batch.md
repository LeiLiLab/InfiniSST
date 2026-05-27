## Hypothesis

The newly trained Japanese TM-SFT model with exact GT term wrapping should improve tagged-ACL terminology accuracy when paired with the HN1024 retriever, while keeping BLEU competitive with the earlier Japanese RASST readout.

## Background / Motivation

This is the Japanese counterpart to the current German tagged-term TM-SFT rescue path.  The model was trained from `20260525T0250__speech_llm_train__tmsft_gttermwrap_exact_ja_r32a32_ep4_taurus8` and exported to Hugging Face format.  We first run `lm=2` on tagged ACL raw with same-lm batch evaluation, then use the result to decide whether to run `lm=1,3,4`.

## What changed vs baseline

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_tmsft_gttermwrap_exact_ja_r32a32_ep4_taurus8/keep1.0_r32/v0-20260525-104902-hf`
- Runtime glossary/eval denominator: `acl6060_tagged_gt_raw_min_norm2`
- Retriever: HN1024 MaxSim checkpoint
- RAG threshold: `tau=0.78`
- Batch eval settings: `max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`, five talks batched for the same latency multiplier.

## Expected metrics

The gate is `lm=2` tagged ACL raw.  Desired behavior is higher TERM_ACC than no-RAG and BLEU that does not collapse relative to the prior Japanese RASST/TM-SFT references.  The script writes a per-run summary TSV under the output `__summary__` directory and validates that both raw and stripped instance logs have five rows.

## Verdict

Pending.  The launcher is submitted on Taurus GPU pair 5,6 and may wait until the pair is idle before starting.
