## Hypothesis

Training the Speech LLM on retriever-generated term maps whose supervision is
defined by source-text exact matches against the imported 100k glossary should be
more faithful to the intended inference setting than V1/V1b, where historical
`gt_terms_by_chunk` was not a stable source of truth.

## Background / Motivation

V1 had glossary/GT mismatch. V1b fixed coverage by unioning historical GT terms
into the retrieval bank, but it still trusted historical GT. This V2 line keeps
the deployed 100k glossary as the glossary source of truth and rebuilds
`gt_terms_by_chunk` from exact source chunk matches before retriever term-map
generation.

## What changed vs baseline

- Parent data event:
  `20260519T1235__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k`
- Training data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519/train_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl`
- Validation data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519/dev_s_zh_retriever_timeline_lh1b88kw_tau073_srcmatch100k_k10_lb1p92.jsonl`
- Base model: initial Qwen3-Omni mcore checkpoint, not the previous pure-streaming HF baseline.
- LoRA: rank 8, alpha 32, one epoch.
- Retriever data policy: timeline MaxSim over `[chunk_start - 1.92s, chunk_end]`,
  tau=0.73, top-k=10, no GT backfill.
- Source-GT policy: whole-token exact matches between TSV source trajectory
  chunks and imported zh100k glossary terms.

## Expected metrics

This run should help determine whether Speech LLM SFT improves when the training
term-map supervision is defined against the same glossary used for retrieval.
Because the 100k glossary includes many common one-word terms and its target
translations may not match reference wording, downstream TERM_ACC may still be
bounded by metric exactness rather than true translation quality alone.

## Verdict

Pending.
