## Hypothesis

Increasing Speech LLM LoRA capacity from rank 8 / alpha 32 to rank 32 / alpha
64 may improve term-map conditioning on the V2 source-match retriever SFT data.
This is a capacity ablation only; data, base model, retriever, tau, top-k, and
training duration are kept identical to the V2 r8/a32 run.

## Background / Motivation

The original pure-streaming baseline used low-rank LoRA.  Earlier term-map runs
used larger rank settings, but it is unclear whether the gain came from capacity
or from data construction.  V2 rebuilt `gt_terms_by_chunk` from exact source
matches against the imported zh100k glossary, so this run isolates LoRA capacity
on the same data policy.

## What changed vs baseline

- Baseline planned run:
  `20260519T1300__speech_llm_train__retriever_timeline_zh_lh1b88kw_tau073_srcmatch100k_r8a32`
- Parent data event:
  `20260519T1235__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k`
- Training data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519/train_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl`
- Validation data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519/dev_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl`
- Base model: initial Qwen3-Omni mcore checkpoint.
- LoRA: rank 32, alpha 64, one epoch.
- Retriever data policy: timeline MaxSim over `[chunk_start - 1.92s, chunk_end]`,
  tau=0.73, top-k=10, no GT backfill.

## Expected metrics

If rank 8 is capacity-limited, this run should improve downstream TERM_ACC and
REAL_TERM_ADOPT under strict medicine and tagged ACL oracle/retriever evals. If
larger LoRA overfits or destabilizes the streaming policy, BLEU or StreamLAAL may
degrade relative to r8/a32 despite similar term metrics.

## Verdict

Pending.
