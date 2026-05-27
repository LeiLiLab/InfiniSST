# Speech LLM De Retriever Cap16 Exact-Boundary, r32a32, Taurus8

## Hypothesis

German Speech LLM fine-tuning with HN1024 retriever-recalled term maps capped to 16 entries and exact assistant-side `<term>` wrapping should recover terminology control without repeating the uncapped TM-SFT BLEU degradation.

## Background / Motivation

The historical De/Ja TM-SFT data used LLM-generated term maps and could expose very dense term maps. The new data-prep event builds a controlled German branch using HN1024 retriever-recalled term maps, GT backfill, cap16 density control, and exact/boundary-only assistant target wrapping.

Parent data event:

`20260525T0348__data_prepare__deja_termmap_ablation_cap16_exactboundary`

## What changed vs baseline

- Training JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/de/retriever_hn1024_tau078_cap16_exactboundary/train_s_de_retriever_hn1024_tau078_cap16_gttermwrap_exactboundary.jsonl`
- Dev JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/de/retriever_hn1024_tau078_cap16_exactboundary/dev_s_de_retriever_hn1024_tau078_cap16_gttermwrap_exactboundary_first355.jsonl`
- Term maps come from HN1024 retriever outputs at tau=0.78 with `tcm_filtered_with_gt_backfill`.
- Each chunk keeps at most 16 term-map entries, preserving GT terms first.
- Assistant targets contain exact `<term>...</term>` wrapping, with only boundary-only repair for adjacent chunk splits.
- Training config follows the existing 8-GPU TM-SFT exact-wrap setting: LoRA `r32a32`, EP=4, TP=1, global batch size 8. The first startup attempt with max length 2048 failed during strict preprocessing because one packed row had length 2385. The retry uses max length 3072, matching the safer NewV10 SFT setting and avoiding silent row drops.

## Expected metrics

First readout should be German tagged ACL raw `lm=2`, HN1024, `tau=0.78`, same-lm batch, `max_new_tokens=80`.

Pass signal:

- BLEU should be competitive with the verified InfiniSST/no-RAG `lm=2` baseline.
- TERM_ACC should remain clearly above no-RAG and ideally approach or exceed the TM-SFT + HN1024 reference.

## Verdict

Submitted on Taurus as an 8-GPU idle-watched training job. First startup attempt failed before W&B initialization with `MaxLengthError: Current length of row(2385) is larger than the max_length(2048)`. Retried with `MAX_LENGTH=3072`. Pending W&B startup, training completion, and HF export.
