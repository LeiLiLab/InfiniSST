# 2026-05-21 varctx LMLB v3 no-HN gc256 resume latest on taurus8

## Hypothesis

Resume the no-hard-negative ablation from the most recent overwrite-only latest
checkpoint and continue the same 8k-global-batch training recipe on taurus 8GPU.
Saving a fresh latest checkpoint after every eval should make repeated pause and
resume cycles operationally safe even when best metrics do not improve.

## Background / Motivation

The prior no-HN resume run `bgz7akb6` was manually paused to release aries GPUs.
Before pause it had written:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_aries_latest.pt`

The latest checkpoint corresponds to the eval checkpoint around step 800 in the
paused run log.  This event continues from that checkpoint on taurus because
aries is currently unavailable.

## What changed vs baseline

- Resume checkpoint: `bgz7akb6` latest checkpoint from aries.
- Compute target: direct detached execution on taurus GPUs `0,1,2,3,4,5,6,7`.
- Hard negatives remain disabled: `hard_neg_k=0`, `hard_neg_k_per_sample=0`.
- Global batch remains 8192 with equal per-rank batch 1024 over 8 GPUs.
- `gradient_cache_chunk_size=256`.
- `save_latest_on_eval=true`.
- `eval_top100_samples=0`.
- `tcm_sweep_thresholds=0.75`.
- Primary best metric: `eval_dev/recall@10_gs10000`.
- Secondary best metric: `eval_acl6060/recall@10`.

The taurus run writes to a separate `_taurus` checkpoint stem so it does not
overwrite the previous aries checkpoint stem.

## Expected metrics

Training should continue from the latest no-HN state rather than restart from the
older best checkpoint.  The expected operational signal is successful WandB init,
resume load, continued train steps, periodic eval, and overwrite-only latest
checkpoint updates after each eval.

## Verdict

Manually stopped on 2026-05-22 after convergence was judged sufficient.  W&B run
`40fgbr2y` is the canonical record.  The selected checkpoint for downstream use
is the secondary-best checkpoint selected by `eval_acl6060/recall@10`:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8192_gc256_wr1000k_m0.0_maxsim_mfa_variantE_nohn_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu01234567_taurus_best_eval_acl6060_recallat10.pt`

The latest overwrite checkpoint was last completed at the prior stable eval
before the manual stop and remains available for resume/debugging.
