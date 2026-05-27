# HN512 resume after scheduled HN256 pause

## Hypothesis

Continuing HN512 from its latest checkpoint should give the midpoint
hard-negative-depth ablation enough training budget for comparison against HN256
and HN1024.

## Background / Motivation

HN512 run `5fwrs7rh` was paused manually after step 320 to free Taurus GPUs for
HN256. This scheduled resume starts only after `gsjheh6r` is paused by the timer.

## What changed vs baseline

- Resume source run: `5fwrs7rh`.
- Resume checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_latest.pt`
- HN setting remains:
  - `hard_neg_k=0`
  - `hard_neg_k_per_sample=512`
  - `grad_cache_chunk_size=256`
  - TCM off
- Checkpoint selection remains dev-primary:
  - primary: `eval_dev/recall@10_gs10000`
  - secondary readout checkpoint: `eval_acl6060/recall@10`
- Compute:
  - GPU list: `0,1,2,3,4,5`
  - effective global batch: `8190 = 6 * 1365`
- Medicine eval uses the current strict clean MFA exact-only medicine dataset.

## Expected metrics

The resumed line should keep the previous best trackers from the checkpoint and
only overwrite best checkpoints if later evals improve the corresponding
metrics.

## Verdict

FAILED after the scheduled resume. The run started as W&B `gasqw118`, resumed
from the HN512 latest checkpoint at step 320, completed the step-400 eval, and
saved the latest checkpoint. Training then hit CUDA OOM on rank 1 during the
post-eval training step, followed by an NCCL watchdog abort. The freshest
checkpoint preserved by this run is the step-400 latest checkpoint.
