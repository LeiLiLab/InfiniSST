## Hypothesis

V3 real-retriever robust SFT should improve Speech LLM stability under empty/sparse term_map chunks and noisy large-glossary term_map chunks relative to the dense V2 retriever-timeline SFT.

## Background / Motivation

Observed failures are Speech LLM robustness failures rather than strict-term recall failures: `de lm3/raw` can enter English-copy mode when early chunks have no term_map, while `ja lm1/gs10k` can over-copy noisy false-positive term_map entries.  V3 reshapes the training term_map distribution to include empty, sparse, clean-GT, realistic, partial-noisy, dense-noisy, and term-critical chunks.

## What changed vs baseline

- Baseline W&B candidate: `sst_omni/dvpmpqma` (`v2_srcm_r8a32`)
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520/train_s_zh_v3_real_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520/dev_s_zh_v3_real_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Data manifest: `20260520T0000__data_prepare__v3_robust_termmap_zh`
- LoRA: rank 8, alpha 32
- Compute: taurus hold job `45269`, 2 GPUs
- Term-map format: legacy plain `source=target`

## Expected metrics

On downstream simuleval, this should reduce early no-term English-copy cascades and reduce over-copy under gs10k without requiring larger LoRA rank.

## Verdict

Pending.
