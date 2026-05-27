# HN depth continuation — `k=4096`, resume from `l4i457ih` best ACL checkpoint

Continue from the `l4i457ih` checkpoint selected by
`best_acl6060_gs10000`. This run keeps the useful checkpoint state from the
earlier `resume3000` attempt but avoids the bad LR schedule by holding a
conservative constant LR.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hn4096_l4best_c5e5_aries8`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_resume3000_l4best_const5e5_8gpu_aries.sh`
- **Baseline/source runs**:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/l4i457ih

## Hypothesis

If the `l4i457ih` best ACL checkpoint captured a useful continuation point
despite the warmup mistake, then resuming it with Adam state and a stable
constant LR should continue improving the `k=4096` TCM-off base without the LR
overshoot risk.

## Background / Motivation

The first `resume3000` run (`l4i457ih`) accidentally resumed from a plain model
checkpoint without optimizer/scheduler state, so LR restarted warmup from near
zero and was rising toward `1.7e-4`. The checkpoint requested here,
`*_best_acl6060_gs10000.pt`, was saved at global step `600` with optimizer state
and LR around `7.98e-5`. To avoid another schedule artifact, this run resumes
that checkpoint and sets a fixed `5e-5` LR.

## What changed vs baseline

- Baseline/source run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/l4i457ih
- Diff:
  - resume checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_resume3000_aries8_smoke3000_best_acl6060_gs10000.pt`
  - resume global step: `600`
  - hparam `constant_lr`: `0` -> `5e-5`
  - hparam `max_steps`: continue to global step `3000`
  - hparam `num_gpus`: `8` (unchanged)
  - hparam `per_gpu_batch`: `768` (unchanged)
  - hparam `batch_size`: `6144` (unchanged)
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - data / HN depth / MFA recipe: unchanged

## Expected metrics

The primary read is dev/train only; ACL6060 is retained as an external one-shot
probe for later interpretation.

- `train/lr` should stay flat at `5e-5`.
- `train/loss_infonce` should not spike after resume.
- `eval_dev/recall@10_gs10000` and `eval_dev/topk10_filtered_recall@tau_0p80_gs10000` should improve or plateau relative to `iaiyi1m8` and the `l4i457ih` checkpoint.
- Fixed-threshold noise should be monitored, but not used alone to reject the base continuation.

## Verdict

CANCELLED on 2026-04-25 after the run showed overfitting/noise pressure and the
cost profile made `k=4096` unsuitable as a sweep point. Keep the saved best
checkpoints only as a fully trained `k=4096` reference; move follow-up TCM/HN
sweeps to cheaper `k=1024` or `k=2048` settings.
