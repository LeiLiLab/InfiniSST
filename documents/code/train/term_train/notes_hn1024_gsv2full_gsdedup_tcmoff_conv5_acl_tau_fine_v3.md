# HN1024 GSv2full gsdedup TCM-off conv5 ACL tau fine readout

## Hypothesis

The dev-only recall-retention calibrated tau candidates should preserve ACL
recall better than the F-score-optimal high-tau settings, while reducing
emissions relative to very low thresholds.

## Background / Motivation

Tau was calibrated on held-out dev distributions, not on ACL. This run is an
ACL-only readout for the candidate thresholds around the dev-selected range, so
ACL metrics are used for reporting and sensitivity analysis only.

## What changed vs baseline

- Checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`
- Dev eval: disabled.
- ACL eval dataset: `/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl`
- ACL glossary: `/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060_gt_union_gs10000.json`
- ACL glossary sizes: raw/base, `1000`, `10000`
- Sweep taus: `0.70` to `0.86` in `0.02` increments.

## Expected metrics

- Primary readout: ACL recall@10 and tau-filtered recall for raw/base, gs1k,
  and gs10k.
- Secondary readout: precision micro/macro, F3, kept avg, and no-term emitted
  avg for the same tau values.
- Candidate tau values of interest: `0.74`, `0.76`, and `0.78`.

## Verdict

Pending. Use only as post-calibration ACL/OOD reporting, not for choosing tau.
