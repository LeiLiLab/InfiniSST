## Hypothesis

V7 refmatch R95 SFT should match the deployed retriever's approximate 95%
strict-term recall better than the near-oracle V6 curriculum, while preserving
reference-compatible exact target supervision.

## Background / Motivation

V6 achieved 99.88% GT-term-in-term-map rate, which is too close to oracle and
may overfit the Speech LLM to perfect term maps.  V7 keeps the same
reference-compatible target filter but drops a controlled subset of GT terms to
simulate realistic retriever misses.

## What changed vs baseline

- Baseline W&B candidate: `sst_omni/dvpmpqma` (`v2_srcm_r8a32`)
- Data manifest: `20260521T1350__data_prepare__v7_refmatch_r95_termmap_zh`
- Training data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v7_refmatch_r95_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/train_s_zh_v7_refmatch_r95_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- Dev data: `/mnt/gemini/data1/jiaxuanluo/speech_llm_v7_refmatch_r95_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260521/dev_s_zh_v7_refmatch_r95_termmap_lh1b88kw_tau073_srcmatch100k.jsonl`
- LoRA: rank 8, alpha 32
- Compute: taurus hold job `45269`, 2 GPUs when available
- Term-map format: legacy plain `source=target`

## Expected metrics

Primary downstream check is tagged ACL `zh lm2/raw`.  V7 should outperform V3
and avoid V6's near-oracle mismatch, ideally narrowing the exact TERM_ACC gap to
no-TM-SFT while keeping robustness under sparse/noisy term maps.

## Verdict

Pending.
