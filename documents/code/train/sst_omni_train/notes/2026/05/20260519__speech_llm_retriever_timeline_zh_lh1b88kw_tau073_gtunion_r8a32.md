## Hypothesis

Training the Speech LLM on GT-union retriever-generated timeline term_maps
should expose it to realistic retriever misses and noise while avoiding the V1
glossary-mismatch artifact.  This first V1b SFT keeps LoRA rank 8 / alpha 32 so
capacity remains comparable to the original pure-streaming baseline.

## Background / Motivation

The first V1 data build used a filler glossary that did not cover many
`gt_terms_by_chunk` entries, so low GT-term recall partly measured missing
glossary entries rather than retriever behavior.  V1b builds a GT-union
glossary from the zh train/dev JSONL plus the zh100k filler glossary, then uses
the same `lh1b88kw` retriever at tau=0.73.

## What changed vs baseline

- Parent data event:
  `20260519T1150__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion`
- Training data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion_20260519/train_s_zh_retriever_timeline_lh1b88kw_tau073_gtunion_k10_lb1p92.jsonl`
- Validation data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion_20260519/dev_s_zh_retriever_timeline_lh1b88kw_tau073_gtunion_k10_lb1p92.jsonl`
- Base model: initial Qwen3-Omni mcore checkpoint, not the previous pure-streaming HF baseline.
- LoRA: rank 8, alpha 32, one epoch.
- Sequence cap: `max_length=4096`.
- Retriever data policy: timeline MaxSim over `[chunk_start - 1.92s, chunk_end]`, tau=0.73, top-k=10, no GT backfill.

## Expected metrics

Primary downstream checks are SimulEval TERM_ACC, REAL_TERM_ADOPT, true
sentence-level TERM_FCR, BLEU, and StreamLAAL on strict medicine and tagged ACL
readouts.  Relative to oracle-GT SFT, this run is expected to have lower
TERM_ACC and higher FCR, but it should be more representative of inference-time
retriever noise.

## Verdict

Pending.
