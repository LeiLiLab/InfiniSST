# Variable-context retriever HN512 ablation, gc256, Taurus 6GPU

## Hypothesis

Using `hard_neg_k_per_sample=512` should test the middle point between the
original HN1024 run and the weaker HN256 / no-HN settings. The intended signal is
whether half-depth hard negatives recover the recall benefit of `lh1b88kw`
without reproducing the stronger tau-filtered precision decline seen in HN1024.

## Background / Motivation

Source run `lh1b88kw` used the balanced 2.88s/3.84s/4.80s/5.76s GSV2-full
GSDedup variable-context dataset with global batch `8192`,
`hard_neg_k_per_sample=1024`, `grad_cache_chunk_size=128`, TCM-off, MaxSim MFA,
and six epochs on Aries. The no-HN line has now been manually stopped and marked
canonical at W&B run `40fgbr2y`; the HN256 line is W&B run `e981df6j`.

## What changed vs baseline

- Source HN run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Comparison run ids: `e981df6j`, `40fgbr2y`, `bgz7akb6`
- Ablation:
  - `hard_neg_k_per_sample`: `1024` -> `512`
  - `grad_cache_chunk_size`: `128` -> `256`
  - GPU list: `0,1,2,3,4,5` on Taurus hold allocation `45269`
  - target batch remains 8k; equal-rank 6GPU training uses `1365 * 6 = 8190`
    effective global batch because this launcher requires the same per-rank
    batch on every process.
- Protocol guardrail:
  - checkpoint selection stays dev-primary: `eval_dev/recall@10_gs10000`.
  - secondary saved metric is `eval_acl6060/recall@10` for readout tracking.
  - ACL remains held-out readout only and is not used to choose tau,
    checkpoint, or variant winner.
  - top-100 per-sample eval logging is disabled.
  - TCM threshold sweep is restricted to `0.75`.

## Expected metrics

HN512 should be interpreted as an HN-depth response curve point. The useful
evidence is whether it lands between HN1024 and no-HN on recall, tau-0.75
precision, and no-term noise, while using the same dev-primary selection rule.
Because this uses six GPUs, the effective batch is `8190`, not exactly `8192`;
record that as a compute constraint in comparisons.

## Verdict

PAUSED manually on 2026-05-22 16:17 UTC to release Taurus GPUs for the HN256
resume. The last completed eval was step `320`; the run was stopped after
resuming training to about step `323`.

Resume checkpoint:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_latest.pt`

Best primary checkpoint observed before pause:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_best.pt`

Best secondary checkpoint observed before pause:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_best_eval_acl6060_recallat10.pt`

Compare against `lh1b88kw`, `e981df6j`, `40fgbr2y`, and `bgz7akb6` using WandB
at-best-step bundles, with ACL treated as held-out readout only.
