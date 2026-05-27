## Hypothesis

Training ja on the rebuilt New V9 MFA source-exact, npfilter source-union, lexexact old-new_v3 data with 8 Taurus GPUs should finish faster than the canceled Aries 4-GPU run while preserving the same data recipe and LoRA capacity.

## Background / Motivation

The Aries 4-GPU ja run was too slow: at iteration 200/723 it still had roughly 6.3 hours remaining. Taurus currently has all eight GPUs idle, so this event replaces the Aries run with an 8-GPU Taurus run.

## What changed vs baseline

- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_20260524/train_s_ja_new_v9_mfa_openai_rewrite_oldnewv3.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_20260524/dev_s_ja_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl`
- Data-prep parent event: `20260524T1552__data_prepare__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja`
- Replaces canceled train event: `20260524T1806__speech_llm_train__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_r32a64_tp2_aries4`
- LoRA: rank 32, alpha 64.
- Runtime: Taurus 8 GPUs, TP=2, EP=2, sequence parallel, one epoch.
- GPU set: physical devices `0,1,2,3,4,5,6,7`.
- Global batch size increases from 4 to 8 to use the 8-GPU run.

## Expected metrics

After export, run tagged ACL raw glossary quick eval with HN1024 tau=0.78 for `lm=1,2,3,4`, using `max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, and stripping `<term>` tags before metrics. This run should replace the polluted old ja New V9 line and the canceled Aries 4-GPU attempt.

## Verdict

Pending training completion and HF export.
