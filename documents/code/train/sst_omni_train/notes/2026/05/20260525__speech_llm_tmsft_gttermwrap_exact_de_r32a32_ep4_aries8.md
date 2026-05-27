# Speech LLM TM-SFT Exact GT Term Wrapping, De, r32a32, Aries8

## Hypothesis

Adding exact assistant-side `<term>` wrapping on top of the historical German
TM-SFT data can improve term controllability while preserving the stronger BLEU
behavior observed from the TM-SFT SLM family.

## Background / Motivation

This branch intentionally avoids the larger NewV9/NewV10 data changes. It uses
the historical German TM-SFT data line and changes only the assistant target
surface when a GT term target translation is exactly supported by future
assistant text.

Parent data event:

`20260525T0045__data_prepare__tmsft_gttermwrap_exact_de`

## What changed vs baseline

- Training JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_tmsft_gttermwrap_exact_de_20260525/train_s_de_tmsft_gttermwrap_exact.jsonl`
- Dev JSONL:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_tmsft_gttermwrap_exact_de_20260525/dev_s_de_tmsft_gttermwrap_exact_first355.jsonl`
- Compared with historical TM-SFT, assistant targets now contain exact
  `<term>...</term>` wrapping for supported GT target translations.
- No retriever rebuild, no LLM variant augmentation, and no no-GT term-map
  zeroing are used.

The launcher keeps the TM-SFT-style LoRA setting `r32a32` and `max_length=2048`,
but runs on Aries 8 GPUs with `global_batch_size=8` because the current
multi-GPU wrapper requires the global batch to be divisible by visible ranks.

## Expected metrics

First readout should be German tagged ACL raw `lm=2`, HN1024, `tau=0.79`,
same-lm batch, `max_new_tokens=80`.

Pass signal:

- BLEU should be at least competitive with verified InfiniSST/no-RAG `lm=2`
  BLEU `30.0676`.
- TERM_ACC should be closer to the TM-SFT + HN1024 reference than to no-RAG.

## Verdict

Pending Aries 8-GPU training completion and HF export.
