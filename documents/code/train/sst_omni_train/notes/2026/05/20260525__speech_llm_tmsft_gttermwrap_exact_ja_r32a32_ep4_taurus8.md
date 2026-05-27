# Speech LLM TM-SFT Exact GT Term Wrapping, Ja, r32a32, Taurus8

## Hypothesis

Japanese TM-SFT plus exact assistant-side `<term>` wrapping should provide a fast tagged-term SLM candidate analogous to the German rescue branch, while preserving the historical TM-SFT input term-map exposure.

## Background / Motivation

This branch intentionally mirrors the current German TM-SFT exact GT term-wrap treatment and avoids the larger NewV9/NewV10 data changes.

Parent data event:
`20260525T0245__data_prepare__tmsft_gttermwrap_exact_ja`

## What changed vs baseline

- Training JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_tmsft_gttermwrap_exact_ja_20260525/train_s_ja_tmsft_gttermwrap_exact.jsonl`
- Dev JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_tmsft_gttermwrap_exact_ja_20260525/dev_s_ja_tmsft_gttermwrap_exact_first355.jsonl`
- Compared with historical Japanese TM-SFT, assistant targets now contain exact `<term>...</term>` wrapping for supported GT target translations.
- No retriever rebuild, no LLM variant augmentation, and no no-GT term-map zeroing are used.

Runtime uses LoRA `r32a32`, `max_length=2048`, EP=4, TP=1, sequence parallel disabled, Taurus 8 GPUs, and one epoch.

## Expected metrics

First readout should be Japanese tagged ACL raw at `lm=2`, HN1024, `tau=0.79`, same-lm batch, `max_new_tokens=80`.

## Verdict

Pending Taurus 8-GPU training completion and HF export.
