## Hypothesis

Training ja on the rebuilt New V9 MFA source-exact, npfilter source-union, lexexact old-new_v3 data should preserve the zh-winning Speech LLM recipe while avoiding the polluted ja GT/fuzzy-match failure mode.

## Background / Motivation

The 6-GPU Aries run was replaced before training because GPUs 6 and 7 were occupied by medicine SimulEval. This run starts immediately on the currently free Aries GPUs 0,1,2,5.

## What changed vs baseline

- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_20260524/train_s_ja_new_v9_mfa_openai_rewrite_oldnewv3.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja_20260524/dev_s_ja_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl`
- Data-prep parent event: `20260524T1552__data_prepare__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_ja`
- LoRA: rank 32, alpha 64.
- Runtime: aries 4 GPUs, TP=2, EP=2, sequence parallel, one epoch.
- GPU set: physical devices `0,1,2,5`.

## Expected metrics

After export, run tagged ACL raw glossary quick eval with HN1024 tau=0.78 for `lm=1,2,3,4`, stripping `<term>` tags before metrics. The run should replace the polluted old ja New V9 line, not be compared as a simple continuation of it.

## Verdict

Pending training completion and HF export.
