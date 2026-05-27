## Hypothesis

For En-De tagged ACL, the historical TM-SFT SLM plus HN1024 at tau=0.78 should provide a focused lm=2 reference point when empty retrieved term maps are omitted.

## Background / Motivation

The Aries submission was waiting for GPUs. This Taurus run uses the idle NVLink pair 4,5 to get the focused de/lm=2 result sooner.

## What changed vs baseline

- Model: `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
- Retriever: HN1024, top-k=10, tau=0.78, lookback=1.92s
- Dataset/glossary: tagged ACL raw, `acl6060_tagged_gt_raw_min_norm2`
- Batch setting: same-lm batch, de/lm=2 only, five talks, max_new_tokens=80
- Runtime prompt policy: `empty_term_map_policy=omit`
- Compute: Taurus GPU pair 4,5

## Expected metrics

Expected to be comparable to the old de/lm=2 TM-SFT+HN1024 row, with possible BLEU/TERM_ACC changes from omitting empty term-map blocks.

## Verdict

Completed with valid eval artifacts. The top-level launcher exited nonzero after the metric TSV was written because the summary-merge helper used an invalid `Path.glob` pattern; this affected summary collation only, not generation or scoring. The launcher has been patched and the summary TSVs were backfilled from the verified `eval_results.tsv`.

Verified metrics from `eval_results.tsv`: BLEU=31.0329, StreamLAAL=1623.5111, StreamLAAL_CA=875.2451, TERM_ACC=0.8235 (770/935). Both `instances.log` and `instances.strip_term.log` contain 5 rows.
