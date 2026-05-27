# New V4 cache-only LLM-variant augmentation on new_v3 data

## Hypothesis

Applying the V16 natural target-translation replacement policy to the older `new_v3` retriever-SFT data may recover the stronger `new_v3` data distribution while adding a direct adoption signal for term-map values.

## Background / Motivation

The V13/V15/V16 line used the newer timeline retriever data, but quick eval did not clearly improve over previous LLM-generated term-map SFT.  The older `new_v3` data had stronger prior results and already contains dense retriever term maps plus GT backfill.  This event keeps the old data distribution and applies the V16 replacement strategy.

## What changed vs baseline

- Input train data:
  `/mnt/gemini/data1/jiaxuanluo/tcm_wiki100k_gt_zh_tau075_d9_k20_postfiltercap_termmap_v2_sourcefinal_gtzh/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_sourcefinal_tcmwiki100kgt_tau075_d9_k20_postfiltercap_gtzhoverride.jsonl`
- Input dev/control data:
  `/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20_final.jsonl`
- Variant source:
  V16 OpenAI cache at `/mnt/gemini/data1/jiaxuanluo/speech_llm_v16_llm_variant_aug_retriever_timeline_zh_lh1b88kw_tau073_20260522/openai_term_variant_cache.json`
- API policy:
  `--cache-only`; no OpenAI calls are made in this event.
- Missing GT policy:
  `--missing-gt-policy keep_unchanged`; legacy rows without `gt_terms_by_chunk` are counted and written unchanged.

## Expected metrics

This data should preserve the older `new_v3` term-map distribution while adding a partial natural-variant adoption signal.  Because cache coverage is incomplete, it is a pilot rather than a full V16-style rewrite.

## Verdict

Pending data preparation.
