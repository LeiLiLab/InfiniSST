# HN depth continuation — `k=4096`, `TCM off`, resume from `iaiyi1m8`, fixed LR schedule

Continue the successful `k=4096`, batch-6k TCM-off scout from its step-400
checkpoint on Aries 8 GPU. This is the corrected continuation after the first
`resume3000` submission restarted LR warmup from near zero.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hn4096_res3000_fixlr_aries8`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_resume3000_fixlr_8gpu_aries.sh`
- **Baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8

## Hypothesis

If the `iaiyi1m8` scout was still under-trained at step 400, continuing the
same `k=4096` TCM-off recipe to `max_steps=3000` should improve or stabilize
dev dense-bank retrieval and make the later TCM curriculum easier to interpret.

## Background / Motivation

TCM-v2 exploration is deferred until the base `k=4096` retriever is closer to
convergence. The first Aries resume run (`l4i457ih`) was cancelled because the
plain `smoke400.pt` checkpoint lacked optimizer/scheduler state, causing the LR
to restart warmup from near zero. This run resumes from the `_epoch_0.pt`
checkpoint, which carries optimizer state, and uses a no-warmup cosine
continuation from the checkpoint LR to step `3000`.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- Diff:
  - resume checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs6k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout_smoke400_epoch_0.pt`
  - hparam `num_gpus`: `8` (unchanged)
  - hparam `per_gpu_batch`: `768` (unchanged)
  - hparam `batch_size`: `6144` (unchanged)
  - hparam `max_steps`: `400` -> `3000` global steps
  - hparam `epochs`: `1` -> `4` so the resumed run has enough epoch loops to reach step `3000`
  - scheduler: start from checkpoint optimizer LR (`~1.33e-4`) and cosine-decay to `max_steps=3000`, with no new warmup
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - data / HN depth / MFA recipe: unchanged

## Expected metrics

The primary read is dev/train only; ACL6060 is retained as an external one-shot
probe for later interpretation.

- `train/lr` should monotonically decay from about `1.33e-4`, not rise toward `1.7e-4`.
- `eval_dev/recall@10_gs10000` and `eval_dev/topk10_filtered_recall@tau_0p80_gs10000` should improve or plateau relative to `iaiyi1m8`.
- `train/loss_infonce` should keep decreasing or flatten without collapse.
- Fixed-threshold noise should be monitored, but not used alone to reject the base continuation.

## Verdict

PENDING: update after the continuation reaches `max_steps=3000`, wall-time stop,
or a clear dev plateau/failure.
