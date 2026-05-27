# HN1024 GSv2full gsdedup TCM-off conv5 tau sweep, dev gs1M

## Hypothesis

A 1M P31 dev glossary is a stronger stress test for threshold calibration than
10k or 100k glossaries. If a tau preserves recall under this large glossary,
it is more defensible as a fixed calibration threshold before reporting OOD
test performance.

## Background / Motivation

ACL6060 is treated as the test/OOD dataset, so it must not be used to choose
the inference tau. This run evaluates only the held-out dev set with a larger
P31 glossary to support a dev-only calibration rule across base, gs10k, gs100k,
and gs1M conditions.

## What changed vs baseline

- Checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`
- Dev eval glossary: `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json`
- Dev glossary sizes: `1000000`
- ACL eval: disabled for calibration.
- Sweep taus: `0.00` to `1.00` in `0.05` increments.
- F-beta: F3 is logged for reference only; tau selection should primarily use
  recall-constrained calibration.

## Expected metrics

- Primary inspection: `eval_dev/topk10_filtered_recall@tau_*_gs1000000`
- Secondary inspection: precision micro/macro, kept avg, and
  `eval_dev/noterm_noise@top10_tau_*_gs1000000`
- Expected useful tau range: `0.70` to `0.85`, depending on the recall/noise
  tradeoff under gs1M.

## Verdict

Pending. Use together with dev base/gs10k/gs100k curves to choose a single
frozen tau before any ACL/OOD reporting.
