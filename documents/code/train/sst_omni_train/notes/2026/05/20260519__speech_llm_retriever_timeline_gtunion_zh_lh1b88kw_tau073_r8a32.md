## Hypothesis

Fine-tuning the Speech LLM on retriever-generated term maps from the
lh1b88kw/tau=0.73 timeline policy should make the model more robust to noisy
term maps than oracle-GT SFT alone.  This first run keeps the baseline LoRA
capacity (`r=8`, `alpha=32`) so the data-policy effect is comparable to the
original streaming SFT setup.

## Background / Motivation

The first V1 retriever-termmap data build was deprecated because the retrieval
glossary did not cover the source JSONL `gt_terms_by_chunk`.  V1b rebuilds the
data with a GT-union glossary while keeping the inference-like timeline policy:
for each existing streaming chunk, retrieve over `[chunk_start - 1.92s,
chunk_end]`, keep overlapping MaxSim evidence windows, apply `tau=0.73`, and do
not backfill GT terms.

V1b QA:

- Train exact GT hit: 74.26% over all GT terms; 82.68% after a simple
  stopword/pronoun-style term-like filter.
- Dev exact GT hit: 80.30% over all GT terms; 82.61% after the same filter.
- No-GT chunks still often receive non-empty term maps, so this is deliberately
  a high-noise retriever-SFT setting, not an oracle upper bound.

## What changed vs baseline

- Baseline streaming model recipe: initial MCore Qwen3-Omni, one-epoch SFT,
  LoRA `r=8`, `alpha=32`.
- Train dataset:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion_20260519/train_s_zh_retriever_timeline_lh1b88kw_tau073_gtunion_k10_lb1p92.jsonl`
- Dev dataset:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion_20260519/dev_s_zh_retriever_timeline_lh1b88kw_tau073_gtunion_k10_lb1p92.jsonl`
- Data manifest:
  `documents/code/train/sst_omni_train/manifests/2026/05/20260519T1150__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion.json`
- Retriever checkpoint parent: `lh1b88kw`.

## Expected metrics

The model should be compared against:

- original streaming-only zh baseline;
- oracle-GT term_map SFT;
- later r32/a64 capacity ablation only if r8/a32 underuses term maps.

Primary eval targets should be TERM_ACC, REAL_TERM_ADOPT, true sentence-level
TERM_FCR, BLEU, and StreamLAAL on medicine/ACL with fixed strict-term metrics.

## Verdict

Pending.
