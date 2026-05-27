# Speech LLM V7 refmatch R95 SFT: zh r8/a32 on aries

## Hypothesis

V7 refmatch R95 SFT should match the deployed retriever's approximate 95%
strict-term recall better than the near-oracle V6 curriculum, while preserving
reference-compatible exact target supervision.

## Background / Motivation

V6 achieved near-perfect GT-term-in-term-map rate, which is too close to oracle.
V7 keeps the same exact-reference target filter but drops a controlled subset of
GT terms to simulate realistic retriever misses.

## What changed vs baseline

- Data manifest: `20260521T1350__data_prepare__v7_refmatch_r95_termmap_zh`
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v7_refmatch_r95_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/train_s_zh_v7_refmatch_r95_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v7_refmatch_r95_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/dev_s_zh_v7_refmatch_r95_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- LoRA: rank 8, alpha 32
- Compute: aries, GPUs 4 and 5
- Term-map format: plain `source=target`

## Expected metrics

Primary downstream check is tagged ACL `zh lm2/raw`.  V7 should improve over V3
retriever-SFT and narrow the exact TERM_ACC gap to no-TM-SFT while keeping
robustness under sparse/noisy term maps.

## Verdict

Pending.
