# V16 no-GT-zero LLM-variant Speech LLM SFT

## Hypothesis

V16 may under-use term maps because no-GT chunks still contain dense false-positive term maps.  Zeroing term maps on no-GT chunks should improve term-map calibration and increase downstream `REAL_ADOPT`.

## Background / Motivation

The base V16 data applies LLM-generated target-translation variants on top of the V13 retriever-timeline data.  Its train split has dense term maps even on chunks with no GT terms: no-GT nonempty term-map rate is `87.75%`.

## What changed vs baseline

- Baseline data: V16 LLM-variant retriever-timeline data.
- New train data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_no_gt_zero_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/train_s_zh_v16_no_gt_zero_llm_variant_aug_tau073_k10_minctx2p88.jsonl`
- New dev/control data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_no_gt_zero_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/dev_s_zh_v16_no_gt_zero_llm_variant_aug_tau073_k10_minctx2p88_first200.jsonl`
- Rule: if `gt_terms_by_chunk[i]` is empty, rewrite the user chunk to `term_map:NONE`; otherwise keep the V16 term map unchanged.
- Train stats: avg term-map entries/chunk drops from `9.05` to `4.63`; no-GT nonempty rate drops from `87.75%` to `0%`; GT chunk nonempty rate stays `97.22%`.
- LoRA: rank `8`, alpha `32`.
- Compute: aries, GPUs `0,1`.

## Expected metrics

Primary downstream check is tagged ACL quick eval on `zh lm=2 raw`, focusing on `TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.  A useful result would improve `REAL_ADOPT` without a large BLEU regression.

## Verdict

Pending training and eval.
