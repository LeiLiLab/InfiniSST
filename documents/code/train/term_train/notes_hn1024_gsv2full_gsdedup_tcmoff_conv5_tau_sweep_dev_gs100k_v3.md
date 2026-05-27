# HN1024 GSv2full gsdedup TCM-off conv5 tau sweep, dev gs100k

## Hypothesis

Dev tau selection should be more realistic with a 100k P31 glossary than with
the ACL 10k glossary, because a larger in-domain glossary better exposes
false-positive emissions before cross-domain ACL comparison.

## Background / Motivation

The first conv5 tau sweep used the ACL 10k glossary for dev and ACL. That makes
the dev distribution too small and too close to ACL, so the domain-shift signal
is less reliable.

## What changed vs baseline

- Purpose: dense inference tau sweep for the 45195 conv5 best checkpoint with a larger P31 dev glossary.
- Checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`
- Dev eval glossary: `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json`
- Dev glossary sizes: `100000`
- ACL reference glossary: `/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json`
- ACL glossary sizes: `1000 10000`
- Sweep taus: `0.00` to `1.00` in `0.05` increments.
- F-beta metrics are no longer emitted; use filtered recall plus micro/macro
  precision and no-term noise directly.

## Expected metrics

- Primary: `eval_dev/topk10_chunk_any_positive_filtered_recall@tau_0p75_gs100000`
- Inspect: `eval_dev/topk10_filtered_recall`, precision micro/macro,
  `noterm_noise@top10`, and kept avg across all tau values for `gs100000`.
- ACL metrics are reference-only for this run.

## Verdict

Pending. Use the generated tau distribution plots to choose whether the 100k
dev glossary materially shifts the preferred tau versus the previous 10k sweep.
