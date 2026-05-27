# HN1024 GSv2full gsdedup TCM-off conv5 fine tau sweep, dev gs10k/100k/1M

## Hypothesis

A fine-grained dev-only tau sweep can justify a fixed inference threshold by
recall retention rather than by F-score. The selected tau should preserve most
unfiltered recall across glossary sizes while reducing no-term emissions.

## Background / Motivation

ACL6060 is treated as the OOD/test set, so it must not be used for threshold
selection. Coarse 0.05 sweeps suggest the useful region is around 0.70 to 0.85,
but the grid is too coarse to choose a paper-facing threshold. This run narrows
the calibration sweep to dev-only 10k, 100k, and 1M glossary settings.

## What changed vs baseline

- Checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`
- Dev eval glossary: `/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample1000000.json`
- Dev glossary sizes: `10000 100000 1000000`
- ACL eval: disabled for calibration.
- Sweep taus: `0.70` to `0.86` in `0.02` increments.
- F3 is logged only as a reference; tau selection should use recall-retention
  constraints plus no-term emission.

## Expected metrics

- Primary inspection: recall retention relative to unfiltered `recall@10` at
  each glossary size.
- Secondary inspection: precision micro/macro, kept avg, and no-term emitted
  avg across tau values.
- Expected useful tau range: `0.74` to `0.82`.

## Verdict

Pending. Use this run to pick a single frozen tau before any ACL/OOD reporting.
