## Hypothesis

Training de on the rebuilt New V9 MFA source-exact, npfilter source-union, lexexact old-new_v3 data should preserve the zh-winning Speech LLM recipe while avoiding the polluted de GT/fuzzy-match failure mode.

## Background / Motivation

The previous de/ja New V9 line was rejected because GT was derived from term-map/fuzzy matches. The new de dataset uses MFA source matching plus old-new_v3 noun/entity candidate gating, exact target-span support, old-new_v3 TCM filler maps, no-GT-zero, and assistant-side `<term>` tags.

## What changed vs baseline

- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524/train_s_de_new_v9_mfa_openai_rewrite_oldnewv3.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524/dev_s_de_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl`
- Data-prep parent event: `20260524T1530__data_prepare__new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de`
- LoRA: rank 32, alpha 64.
- Runtime: taurus 8 GPUs, TP=2, EP=2, sequence parallel, one epoch.

## Expected metrics

After export, run tagged ACL raw glossary quick eval with HN1024 tau=0.78 for `lm=1,2,3,4`, stripping `<term>` tags before metrics. The run should replace the polluted old de New V9 line, not be compared as a simple continuation of it.

## Verdict

Pending training completion and HF export.
