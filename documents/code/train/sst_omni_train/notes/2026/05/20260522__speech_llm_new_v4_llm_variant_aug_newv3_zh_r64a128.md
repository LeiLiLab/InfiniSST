# New V4 LLM-variant augmentation on new_v3 Speech LLM SFT

## Hypothesis

New V4 should preserve the stronger old `new_v3` retriever-SFT distribution while adding a direct natural-variant adoption signal.  If V13/V15/V16 underperformed because the newer timeline data distribution was worse, this run should be closer to the old `new_v3` line and may improve `REAL_ADOPT` without the artificial V15 marker shift.

## Background / Motivation

The `new_v3` data used dense retriever term maps with GT backfill and previously looked stronger than the newer V13 timeline data.  V16 generated natural Chinese target variants and replaced exact assistant substrings, but was applied to the V13 line.  This run applies the same replacement idea to `new_v3`.

## What changed vs baseline

- Baseline train recipe:
  `documents/code/train/sst_omni_train/run_speech_llm_new_v3_r64a128_4gpu_taurus.sh`
- Train data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v4_llm_variant_aug_newv3_zh_20260522/train_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl`
- Dev/control data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v4_llm_variant_aug_newv3_zh_20260522/dev_s_zh_new_v4_llm_variant_aug_newv3_cacheonly.jsonl`
- Data-prep event:
  `20260522T1235__data_prepare__new_v4_llm_variant_aug_newv3_cacheonly_zh`
- Variant policy:
  V16 OpenAI cache-only natural replacement.
- Cache coverage:
  `11429 / 50711` selected train terms were replaced.
- Legacy missing-GT rows:
  `182` train rows and all `355` dev rows were explicitly counted and written unchanged where needed.
- LoRA:
  rank `64`, alpha `128`, matching the old `new_v3` 4-GPU recipe.

## Expected metrics

The first downstream check is tagged ACL quick eval on `zh lm=2 raw`.  We care most about `TERM_ACC` and `REAL_ADOPT`, while watching BLEU for regression.

## Verdict

Pending training and eval.
