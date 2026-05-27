## Hypothesis

The Japanese exact GT-term-wrapped TM-SFT SLM should improve tagged-ACL raw terminology when used as the RASST generator with the HN1024 retriever.  This run reads out all latency multipliers with one same-lm batch process per multiplier.

## Background / Motivation

The model comes from `20260525T0250__speech_llm_train__tmsft_gttermwrap_exact_ja_r32a32_ep4_taurus8` / W&B `juqlpjdc`.  The previous Taurus lm=2 waiter was superseded by this Aries all-lm readout so that `lm=1,2,3,4` can run concurrently on NVLink 2-GPU pairs.

## What changed vs baseline

- Model: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_tmsft_gttermwrap_exact_ja_r32a32_ep4_taurus8/keep1.0_r32/v0-20260525-104902-hf`
- Runtime glossary and scoring denominator: `acl6060_tagged_gt_raw_min_norm2`
- Retriever: HN1024 MaxSim checkpoint
- RAG threshold: `tau=0.78`
- Batch settings: `max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, `VLLM_MAX_MODEL_LEN=12288`, `max_num_seqs=5`, five talks per lm.
- Aries GPU pairs: `0,1`, `2,3`, `4,5`, `6,7`.

## Expected metrics

Each lm should emit exactly one `eval_results.tsv`, plus `instances.log` and `instances.strip_term.log` with five rows.  The launcher prints a `RESULT` line as soon as each lm finishes and writes a merged summary TSV under `__summary__`.

## Verdict

Pending.
