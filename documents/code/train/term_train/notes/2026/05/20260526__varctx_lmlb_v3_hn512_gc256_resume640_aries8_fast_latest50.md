# HN512 Fast Resume From Step 640 On Aries 8 GPUs

## Hypothesis

Continuing HN512 from the `bkcnqlg9` step-640 latest checkpoint on all 8 Aries GPUs should improve wall-clock throughput while preserving the effective 8k global batch and the existing checkpoint-selection state.

## Background / Motivation

The HN512 ablation was paused on Taurus after W&B run `bkcnqlg9` saved the step-640 latest checkpoint:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt`

The 2026-05-24 direct Aries attempt reached torchrun initialization but was killed by SIGTERM before a W&B run id or new latest checkpoint was recorded, so this event is a fresh Slurm-submitted resume attempt.

## What changed vs baseline

- Resume source: `bkcnqlg9` step-640 latest checkpoint.
- Compute: Aries partition, 8 GPUs.
- Batch: `BATCH_SIZE=8192`, `PER_GPU_BATCH=1024`, 8 ranks.
- GradCache: `GRAD_CACHE_CHUNK_SIZE=256`.
- Eval cadence: `EVAL_STEPS_SAMPLE=100`.
- Latest checkpoint cadence: `SAVE_LATEST_STEPS=50`, overwriting `_latest.pt`.
- Storage: logs and W&B files go under `/mnt/gemini/home/jiaxuanluo` because `/mnt/gemini/data1` is effectively full.
- Scheduler and best trackers are preserved on resume.
- ACL remains readout-only and must not select tau, checkpoint, hyperparameters, or ablation winner.

## Expected metrics

The main success criterion is stable continuation from step 640 without OOM or metric reset. Because eval now happens every 100 steps, the next domain readout after step 640 should occur at step 700. The periodic latest checkpoint should refresh at steps 650, 700, 750, and so on even if best metrics do not improve.

## Verdict

SUBMITTED and waiting for Aries resources. Slurm job `45309` is pending with reason `(Resources)` because Aries currently has other jobs on the node and `nvidia-smi` showed all 8 GPUs occupied. No W&B run id exists yet; startup verification is still pending.
