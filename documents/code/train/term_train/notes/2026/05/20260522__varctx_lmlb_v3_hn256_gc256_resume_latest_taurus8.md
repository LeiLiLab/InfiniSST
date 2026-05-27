# Variable-context retriever HN256 resume, gc256, Taurus 8GPU

## Hypothesis

Continuing the HN256 ablation from the latest checkpoint should test whether the
middle hard-negative setting keeps improving after the paused step-800 state.
Using all eight Taurus GPUs restores the exact `8192` global batch while keeping
the HN256 training objective unchanged.

## Background / Motivation

The HN256 run `e981df6j` was paused on Aries after the last completed eval at
step `800` to release GPUs. Its latest checkpoint is resumeable and was saved
with `save_latest_on_eval=true`. The HN512 run `5fwrs7rh` was paused on Taurus
so that this HN256 continuation can use all eight local GPUs.

## What changed vs baseline

- Resume source run: `e981df6j`
- Resume checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_aries_latest.pt`
- Hard-negative setting remains:
  - `hard_neg_k=0`
  - `hard_neg_k_per_sample=256`
  - `grad_cache_chunk_size=256`
  - TCM off
- Compute changes:
  - GPU list: `0,1,2,3,4,5,6,7` on Taurus hold allocation `45269`
  - exact global batch: `8192 = 8 * 1024`
- Protocol guardrail:
  - checkpoint selection stays dev-primary: `eval_dev/recall@10_gs10000`.
  - secondary saved metric is `eval_acl6060/recall@10` for readout tracking.
  - ACL remains held-out readout only and is not used to choose tau,
    checkpoint, or variant winner.
  - top-100 per-sample eval logging is disabled.
  - TCM threshold sweep is restricted to `0.75`.
  - latest checkpoint is overwritten after every eval for future interruption
    and resume.

## Expected metrics

The useful signal is whether continuing past the paused step-800 state improves
dev recall without moving the tau-0.75 precision/noise trend toward the HN1024
failure mode. The run should be compared against `lh1b88kw`, `e981df6j`,
`5fwrs7rh`, `40fgbr2y`, and `bgz7akb6` using WandB at-best-step bundles.

## Frozen checkpoints

- 2026-05-23 UTC: froze the current secondary-best checkpoint before later
  training can overwrite the rolling best/latest paths.
- Frozen checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/lrdx14pm_hn256_bestsec_acl6060r10_0p9924_step1200_tie1280_frozen_20260523.pt`
- Sidecar metadata:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/lrdx14pm_hn256_bestsec_acl6060r10_0p9924_step1200_tie1280_frozen_20260523.pt.json`
- Metric: `eval_acl6060/recall@10=0.9924`.
- W&B `best_secondary/step` is `1200`; step `1280` tied the same metric value
  (`0.9924`) but did not overwrite the best-secondary checkpoint because the
  training script saves that file only on strict improvement. The step-1280
  rolling latest checkpoint had already been overwritten by step `1360` when
  this backup was made.

## Verdict

INTERRUPTED / SUPERSEDED: manually stopped on 2026-05-23 UTC after the user
requested a restart from the frozen step-1200 checkpoint with different
checkpoint metrics. The frozen checkpoint remains:
`/mnt/gemini/home/jiaxuanluo/train_outputs/lrdx14pm_hn256_bestsec_acl6060r10_0p9924_step1200_tie1280_frozen_20260523.pt`.
The replacement event is
`20260523T0335__retriever_train__varctx_lmlb_v3_hn256_step1200_aclmetric_reset_taurus6`.
