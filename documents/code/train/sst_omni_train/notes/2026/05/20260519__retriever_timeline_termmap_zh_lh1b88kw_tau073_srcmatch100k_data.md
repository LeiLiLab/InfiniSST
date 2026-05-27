## Hypothesis

Rebuilding `gt_terms_by_chunk` from source-text exact matches against the imported
100k glossary should remove the glossary/GT mismatch artifact in V1/V1b and give
a cleaner training dataset for retriever-generated Speech LLM term maps.

## Background / Motivation

The previous V1 dataset measured GT recall against historical
`gt_terms_by_chunk`, which did not necessarily share the same vocabulary as the
retrieval glossary.  V1b fixed coverage by building a GT-union glossary, but that
still trusted the historical GT field.  This V2 data event instead treats the
100k glossary plus source trajectory as the source of truth: source terms are
whole-token exact matches in the streaming source chunk.

## What changed vs baseline

- Input train JSONL:
  `/mnt/gemini/data/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
- Input dev JSONL:
  `/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl`
- Source text authority:
  `train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv` and
  `dev_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv`
- Glossary:
  `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000_zh100k_train_gt.json`
- Retriever:
  `lh1b88kw` tau=0.73 timeline retrieval, top-k=10, lookback=1.92s.
- No GT backfill.  Retrieved exact source-term matches use the glossary
  translation.

## Expected metrics

Compared with V1b GT-union, exact GT recall should be interpretable against the
same imported 100k glossary.  The data may still be noisy because the 100k
glossary contains many common one-word entries, and because glossary target
translations may not match the reference wording exactly.

## Verdict

Succeeded.  The final clean run wrote 12,500 train rows and 355 dev rows with no
dropped rows and no partial files.  Source-match GT uses the imported zh100k
glossary directly: train has 172,726 exact source-term matches over 68,705
streaming chunks.  Retriever timeline term-map generation reached GT recall
0.7705 on train and 0.7440 on dev at tau=0.73/top-k=10, with average term-map
sizes around 9.6 entries per chunk.  This is now a realistic high-noise
retriever-SFT dataset rather than the earlier glossary/GT mismatch artifact.
