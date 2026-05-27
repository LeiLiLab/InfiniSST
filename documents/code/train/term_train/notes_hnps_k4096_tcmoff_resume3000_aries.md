# HN depth continuation — `k=4096`, `TCM off`, resume from `iaiyi1m8`, 8 GPU Aries

Continue the successful `k=4096`, batch-6k TCM-off scout from its step-400
checkpoint on Aries 8 GPU. This run is intended to train the base retriever
closer to convergence before any TCM curriculum is attempted.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hn4096_resume3000_tcmoff_aries8`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_resume3000_8gpu_aries.sh`
- **Baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8

## Hypothesis

If the `iaiyi1m8` scout was still under-trained at step 400, continuing the
same `k=4096` TCM-off recipe to `max_steps=3000` should improve or stabilize
dev dense-bank retrieval and make the later TCM curriculum easier to interpret.

## Background / Motivation

TCM-v2 exploration is deferred until the base `k=4096` retriever is closer to
convergence. Directly starting a TCM curriculum from the 400-step scout would
mix base retriever maturation with the TCM effect. Aries currently has an 8-GPU
slot, so this continuation keeps the original `8 x 768 = 6144` batch and lets
the resumed optimizer/scheduler state control the LR path instead of using a
constant-LR override.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - resume checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout_smoke400.pt`
  - hparam `num_gpus`: `8` (unchanged)
  - hparam `per_gpu_batch`: `768` (unchanged)
  - hparam `batch_size`: `6144` (unchanged)
  - hparam `max_steps`: `400` -> `3000` global steps
  - hparam `epochs`: `1` -> `4` so the resumed run has enough epoch loops to reach step `3000`
  - hparam `constant_lr`: unchanged at `0`; restore optimizer/scheduler state from the checkpoint
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - data / HN depth / MFA recipe: unchanged

## Expected metrics

The primary read is dev/train only; ACL6060 is retained as an external one-shot
probe for later interpretation.

- `eval_dev/recall@10_gs10000` and `eval_dev/topk10_filtered_recall@tau_0p80_gs10000` should improve or plateau relative to `iaiyi1m8`.
- `train/loss_infonce` should keep decreasing or flatten without collapse.
- Fixed-threshold noise should be monitored, but not used alone to reject the base continuation.
- If dev metrics clearly plateau before step `3000`, this run becomes the base checkpoint for later TCM curriculum.

## Verdict

PENDING: update after the continuation reaches `max_steps=3000`, wall-time stop,
or a clear dev plateau/failure.
