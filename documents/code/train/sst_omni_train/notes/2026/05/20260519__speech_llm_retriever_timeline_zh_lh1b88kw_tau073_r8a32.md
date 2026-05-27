## Hypothesis

Training the Speech LLM with retriever-generated timeline term_maps should make
it more robust to realistic term_map noise than the oracle-GT SFT data.  The
first V1 run uses LoRA rank 8 / alpha 32 so it remains comparable to the
original pure-streaming SFT baseline capacity.

## Background / Motivation

The oracle-GT SFT run (`3h4wm92o`) showed that the model can use provided
terminology, but oracle data does not expose inference-time misses and noise.
This run trains from the initial Qwen3-Omni mcore checkpoint using the V1
timeline retriever term_map dataset generated from `lh1b88kw` at tau=0.73.

## What changed vs baseline

- Training data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_20260519/train_s_zh_retriever_timeline_lh1b88kw_tau073_k10_lb1p92.jsonl`
- Validation data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_20260519/dev_s_zh_retriever_timeline_lh1b88kw_tau073_k10_lb1p92.jsonl`
- Parent data manifest:
  `documents/code/train/sst_omni_train/manifests/2026/05/20260519T1114__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073.json`
- Base model: initial Qwen3-Omni mcore checkpoint, not the previous pure-streaming HF baseline.
- LoRA: rank 8, alpha 32, one epoch.
- Sequence cap: `max_length=4096`, matching the general Speech LLM training wrapper default; an earlier `3072` attempt failed strict preprocessing on an overlength sample and is recorded as an invalid pre-W&B attempt.
- Retriever data policy: timeline MaxSim over `[chunk_start - 1.92s, chunk_end]`, tau=0.73, top-k=10, no GT backfill.

## Expected metrics

Primary downstream checks are SimulEval TERM_ACC, REAL_TERM_ADOPT, true
sentence-level TERM_FCR, BLEU, and StreamLAAL on strict medicine and tagged ACL
readouts.  Relative to oracle-GT SFT, this run is expected to have lower
TERM_ACC and higher FCR, but should be more representative of inference-time
retriever noise.

## Verdict

Invalid attempt.  This run was started against the deprecated V1 retriever
term_map data before the data QA issue was caught.  It was cancelled before a
W&B run was created, and the launcher now fails fast unless
`ALLOW_DEPRECATED_V1_SFT=1` is explicitly set.  Do not use this as a valid SFT
run.
