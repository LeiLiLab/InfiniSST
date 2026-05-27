# New V4 LLM-variant augmentation on old new_v3 r32/a64

## Hypothesis

Applying the New V4 natural target-translation variant augmentation to the old
`new_v3` retriever-SFT data distribution, while keeping the proven r32/a64
capacity line, should test whether the augmentation improves term adoption
without the cost and confound of the r64/a128 run.

## Background / Motivation

The old `new_v3` r32/a64 checkpoint is the strongest verified retriever-SFT
line so far on tagged ACL `zh lm=2 raw`: it keeps high BLEU and recovers most of
the no-TM-SFT TERM_ACC while greatly reducing false-copy rate.  The cancelled
`n842mhkv` run used the same New V4 data but launched the r64/a128 4-GPU recipe,
which was not the intended capacity setting.

## What changed vs baseline

- Data-prep event:
  `20260522T1235__data_prepare__new_v4_llm_variant_aug_newv3_cacheonly_zh`
- Train data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v4_llm_variant_aug_newv3_zh_20260522/train_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl`
- Dev/control data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v4_llm_variant_aug_newv3_zh_20260522/dev_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl`
- Capacity:
  LoRA rank `32`, alpha `64`.
- Compute:
  Aries 2 GPUs, EP=2, TP=1, sequence parallel disabled.
- Cancelled wrong launch:
  `sst_omni/n842mhkv` was r64/a128 and should not be used as the intended
  New V4 r32/a64 comparison.

## Expected metrics

Primary quick check is tagged ACL `zh lm=2 raw`, with both strict raw glossary
and one-paper `2022.acl-long.110` extracted glossary readouts.  The useful
signal is higher `TERM_ACC` / `REAL_ADOPT` than old `new_v3` r32/a64 without a
large BLEU regression or FCR increase.

## Verdict

Pending training and eval.
