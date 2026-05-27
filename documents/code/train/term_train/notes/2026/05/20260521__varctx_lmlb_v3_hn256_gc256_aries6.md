# Variable-context retriever HN256 ablation, gc256, Aries 6GPU

## Hypothesis

Reducing per-sample hard negatives from `1024` to `256` should test whether the
precision drop observed in the full-HN run comes from excessive hard-negative
pressure rather than from the variable-context recipe itself. HN256 may keep part
of the recall benefit of `lh1b88kw` while reducing the tau-filtered precision
cost relative to HN1024.

## Background / Motivation

Source run `lh1b88kw` used the balanced 2.88s/3.84s/4.80s/5.76s GSV2-full
GSDedup variable-context dataset with global batch `8192`,
`hard_neg_k_per_sample=1024`, `grad_cache_chunk_size=128`, TCM-off, MaxSim MFA,
and six epochs on Aries. The no-HN ablation later used the same data and
selection protocol with hard negatives disabled. This run is the second HN-depth
ablation point between those endpoints.

## What changed vs baseline

- Source HN run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- No-HN comparison run ids: `bgz7akb6`, `40fgbr2y`
- Ablation:
  - `hard_neg_k_per_sample`: `1024` -> `256`
  - `grad_cache_chunk_size`: `128` -> `256`
  - GPU list: `0,1,2,3,4,5` on Aries hold allocation `45290`
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

HN256 should be interpreted as a training-progress and calibration-shape
ablation rather than a final winner by itself. The desired signal is recall
closer to HN1024 than no-HN, with tau-0.75 precision/noise behavior closer to
no-HN than HN1024. Because this uses six GPUs, the effective batch is `8190`,
not exactly `8192`; record that as a compute constraint in comparisons.

## Verdict

PAUSED manually on 2026-05-22 12:20 UTC to release Aries GPUs. The last
completed eval was step `800`; the run was stopped cleanly after resuming
training to about step `809`.

Resume checkpoint:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_aries_latest.pt`

Best secondary checkpoint observed before pause:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn256_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_aries_best_eval_acl6060_recallat10.pt`

Compare against `lh1b88kw`, `bgz7akb6`, and `40fgbr2y` using WandB
at-best-step bundles, with ACL treated as held-out readout only.
