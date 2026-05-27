# GSDedup TCM-off conv5 inference tau sweep

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa_gsdedup_tau_sweep_eval` / `eval`
- **Variant tag**: `tau_sweep_gsdedup_conv5`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_tcmoff_conv5_tau_sweep_acl_dev_v3.sh`
- **Checkpoint source run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7xu2b4so
- **Checkpoint**: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv5_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`

## Hypothesis

A dense inference-threshold sweep should reveal a tau region that preserves
most dev recall while reducing no-term emissions enough for Speech LLM
post-filtering. F-beta metrics are no longer emitted; selection should use
filtered-recall retention first, with precision/noise as constraints.

## Background / Motivation

The gsdedup conv5 run improved dev gs10000 recall, but the deployment threshold
should be chosen from the inference-time tradeoff between recall, micro
precision, and no-term emitted average. The sweep evaluates tau values from
`0.00` to `1.00` in `0.05` increments on dev, with ACL6060 logged as a
cross-domain reference.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7xu2b4so
- Diff:
  - checkpoint: gsdedup conv5 primary best checkpoint
  - tau grid: `0.00, 0.05, ..., 1.00`
  - reported metrics: filtered recall, micro/macro precision, avg kept if pass, no-term emitted average
  - eval data: dev-v3 no-term data as the primary sweep target; ACL6060 raw 1k/10k as reference

## Expected metrics

The selected tau should preserve dev filtered recall while avoiding a large
no-term emitted average. If ACL6060 prefers a nearby but not identical
tau, dev should remain the primary choice and ACL should be treated as a
generalization check.

## Verdict

PENDING: update after the tau sweep finishes.
