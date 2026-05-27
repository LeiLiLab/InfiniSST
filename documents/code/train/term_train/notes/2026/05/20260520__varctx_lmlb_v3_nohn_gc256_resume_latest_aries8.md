# Variable-context no-HN gc256 resume with latest-on-eval saves

## Hypothesis

Continuing the no-HN ablation from the step-240 checkpoint should complete the
same `lh1b88kw`-anchored comparison without relying on primary or secondary best
metric improvements for checkpoint freshness.

## Background / Motivation

The no-HN taurus run `5dtpt842` reached step 240 before it was manually
cancelled to free GPUs. It wrote:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_aries_best.pt`

Because later resumes may not improve the configured best metrics immediately,
this run enables `--save_latest_on_eval`, which overwrites the same latest
checkpoint after every eval.

## What changed vs baseline

- Resume from W&B run `5dtpt842` step-240 primary best checkpoint.
- Keep no-HN ablation settings:
  - `hard_neg_k=0`
  - `hard_neg_k_per_sample=0`
  - `grad_cache_chunk_size=256`
  - global batch `8192` on 8 GPUs.
- New checkpoint freshness behavior:
  - `SAVE_LATEST_ON_EVAL=true`
  - latest checkpoint path uses the existing save stem with `_latest.pt`.
- Metric tracking:
  - primary `eval_dev/recall@10_gs10000`
  - secondary `eval_acl6060/recall@10` by user request.

## Expected metrics

The run should continue the no-HN ablation training line and write a fresh
resumeable latest checkpoint after every eval, even if neither best metric
improves.

## Verdict

Paused manually on 2026-05-21 01:10 UTC to release aries GPUs. The run had
written an overwrite-only latest checkpoint at eval step 800:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_aries_latest.pt`

Last observed train log before interruption was step 820. GPUs were verified
free after stopping the process.
