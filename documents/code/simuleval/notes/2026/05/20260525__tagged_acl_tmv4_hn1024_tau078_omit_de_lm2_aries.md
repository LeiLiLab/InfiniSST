## Hypothesis

For En-De tagged ACL, the historical TM-SFT SLM plus HN1024 at tau=0.78 should provide a focused lm=2 reference point when empty retrieved term maps are omitted.

## Background / Motivation

The previously verified TM-SFT+HN1024 de/lm=2 row used the older empty-term-map behavior. This run repeats the same de/lm=2 batch setup with `empty_term_map_policy=omit`.

## What changed vs baseline

- Model: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
- Retriever: HN1024, top-k=10, tau=0.78, lookback=1.92s
- Dataset/glossary: tagged ACL raw, `acl6060_tagged_gt_raw_min_norm2`
- Batch setting: same-lm batch, de/lm=2 only, five talks, max_new_tokens=80
- Runtime prompt policy: `empty_term_map_policy=omit`

## Expected metrics

Expected to be comparable to the old de/lm=2 TM-SFT+HN1024 row, with possible BLEU/TERM_ACC changes from omitting empty term-map blocks.

## Verdict

Pending. Validate from `eval_results.tsv`, `instances.log`, `instances.strip_term.log`, and `summary_de_lm2.tsv`.
