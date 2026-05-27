## Hypothesis

Retriever-generated term_map SFT data must use a retrieval bank that covers the
source training JSONL `gt_terms_by_chunk`.  Otherwise low GT-term recall is a
glossary mismatch artifact rather than a retriever or data-policy signal.

## Background / Motivation

The first V1 build used the existing zh100k filler glossary directly.  QA showed
that many source JSONL GT terms were absent or surface-mismatched against that
bank, producing an invalid 47.20% train GT-term recall statistic.  V1b keeps the
same timeline retrieval policy but first builds a GT-union glossary from the
train/dev JSONL and the zh100k filler glossary.

Reference retriever background:
`documents/code/train/term_train/reports/20260518_lh1b88kw_tau073_retriever_readout_for_speech_llm.md`.

## What changed vs baseline

- Baseline V1 data event:
  `20260519T1114__data_prepare__retriever_timeline_termmap_zh_lh1b88kw_tau073`
- Train data:
  `/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
- Dev data:
  `/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl`
- Filler glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json`
- V1b glossary:
  `/mnt/gemini/data1/jiaxuanluo/speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_gtunion_20260519/gt_union_plus_zh100k_glossary.json`
- Retriever checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Retrieval policy is unchanged from V1: `[chunk_start - 1.92s, chunk_end]`,
  overlapping MaxSim windows only, `top_k=10`, `tau=0.73`, no GT backfill.

## Expected metrics

Before any SFT launch, check:

- GT-union glossary audit has near-complete exact coverage of translated
  `gt_terms_by_chunk`;
- train/dev rows and chunks are retained without silent drops;
- GT-term recall is no longer dominated by absent glossary entries;
- no-GT chunk term_map density is reported separately from GT-hit recall.

## Verdict

Completed on Taurus hold `45269`.  V1b wrote 12,500 train rows / 68,705
streaming chunks and 355 dev rows / 891 chunks.  GT-union removed the main
glossary mismatch: exact GT hit is 74.26% on train and 80.30% on dev; a simple
term-like slice excluding obvious stopword/pronoun-style singletons is 82.68%
on train and 82.61% on dev.  No-GT chunks remain high-noise
(`93.35%` train, `94.86%` dev non-empty term_map), so this data is valid only
as high-noise retriever-SFT, not as an oracle upper bound.
