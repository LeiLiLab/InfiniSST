# V15 marker-augmented retriever term-map SFT

## Hypothesis

V15 should increase term-map adoption by making a subset of retrieved GT term
translations necessary for minimizing SFT loss.  If the bottleneck is
`REAL_ADOPT`, this should improve tagged ACL `TERM_ACC` and `REAL_ADOPT`
relative to V13.

## Background / Motivation

V13 uses inference-aligned retriever term maps, but its term-map values are
mostly normal translations.  The Speech LLM can often produce fluent references
without copying from `term_map`, especially for common GigaSpeech terms.  V15
keeps the V13 retrieval policy and injects a stronger adoption signal by
marking about half of the retrieved GT target translations and replacing the
same exact substring in the assistant reference.

## What changed vs baseline

- Baseline data: V13 lm1..6 retriever timeline data.
- Training data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v15_marker_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/train_s_zh_v15_marker_aug_tau073_k10_minctx2p88.jsonl`
- Dev/control data:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_v15_marker_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/dev_s_zh_v15_marker_aug_tau073_k10_minctx2p88_first200.jsonl`
- Marker policy: `{translation}__tm{code}`.
- Augmented train terms: `16781`, about `50.03%` of retrieved GT terms.
- LoRA: rank `8`, alpha `32`.
- Compute: aries, GPUs `6,7`.

## Expected metrics

Primary downstream check is tagged ACL quick eval on `zh lm=2 raw`, focusing on
`TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.  Marker augmentation
may improve adoption but could hurt fluency if the model overfits to artificial
suffixes.

## Verdict

Pending training.
