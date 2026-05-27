# HN depth continuation — `k=4096`, `TCM off`, resume from `iaiyi1m8`, 6 GPU Taurus

Continue the successful `k=4096`, batch-6k TCM-off scout from its step-400
checkpoint on Taurus. This run is intended to finish the base retriever before
any TCM curriculum is attempted.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hn4096_resume900_tcmoff_taurus6`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_resume900_6gpu_taurus.sh`
- **Baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8

## Hypothesis

If the `iaiyi1m8` scout was still under-trained at step 400, continuing the
same `k=4096` TCM-off recipe should improve dev dense-bank retrieval and expose
whether the fixed-threshold noise is a transient calibration issue or a stable
trade-off of the stronger HN recipe.

## Background / Motivation

TCM-v2 exploration is deferred until the base `k=4096` retriever is closer to
convergence. Directly starting a TCM curriculum from the 400-step scout would
mix base retriever maturation with the TCM effect. This run creates a cleaner
TCM-off continuation baseline on the available Taurus 6-GPU slot.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - resume checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout_smoke400.pt`
  - compute: Aries 8 GPU -> Taurus 6 GPU
  - hparam `num_gpus`: `8` -> `6`
  - hparam `per_gpu_batch`: `768` (unchanged)
  - hparam `batch_size`: `6144` -> `4608`
  - hparam `max_steps`: `400` -> `900` global steps
  - hparam `constant_lr`: `0` -> `1e-4` to avoid a scheduler shock across the resume boundary
  - hparam `epochs`: `1` -> `2` so the resumed run enters one additional epoch loop
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - data / HN depth / MFA recipe: unchanged

## Expected metrics

The primary read is dev/train only; ACL6060 is retained as an external one-shot
probe for later interpretation.

- `eval_dev/recall@10_gs10000` and `eval_dev/topk10_filtered_recall@tau_0p80_gs10000` should improve or plateau relative to `iaiyi1m8`.
- `train/loss_infonce` should continue decreasing without a sharp LR or resume discontinuity.
- Fixed-threshold noise should be monitored, but not used alone to reject the base continuation.

## Verdict

SUPERSEDED before start: SLURM `43963` was cancelled while still pending. The
continuation moved to an Aries 8-GPU `max_steps=3000` launcher so it can keep
the original per-GPU batch and resume the checkpoint optimizer/scheduler state
without a constant-LR override.
