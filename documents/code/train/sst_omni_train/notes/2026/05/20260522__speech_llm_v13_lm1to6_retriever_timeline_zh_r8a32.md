## Hypothesis

V13 uses inference-aligned retriever term maps on `lm=1..6` chunks.  It should
reduce the train/inference mismatch from long `lm>6` chunks and provide a
cleaner comparison against earlier retriever-SFT runs.

## Background / Motivation

The previous dense/top10 and precision-filtered SFT variants did not improve
REAL_ADOPT enough.  V13 keeps the real retriever path, but matches the current
agent policy more closely: low-latency chunks wait until enough timeline
context is available, while chunks already at least `2.88s` are retrieved
directly without lookback.

## What changed vs baseline

- Training data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_20260522/train_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88.jsonl`
- Validation/control data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v13_lm1to6_retriever_timeline_zh_lh1b88kw_tau073_20260522/dev_s_zh_v13_lm1to6_retriever_timeline_tau073_k10_minctx2p88_first200.jsonl`
- Retriever: `lh1b88kw` best ACL6060 checkpoint.
- Glossary: zh 100k train glossary.
- Term-map policy: `tau=0.73`, `top_k=10`, no GT backfill.
- LoRA: rank `8`, alpha `32`.
- Compute: taurus, GPUs `6,7`.
- Save root: `/mnt/aries/data7/jiaxuanluo/slm` because `/mnt/gemini/data2`
  is full.

## Expected metrics

Primary downstream check is tagged ACL quick eval on `zh lm=2 raw`, focusing on
`TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.

## Verdict

Pending training.
