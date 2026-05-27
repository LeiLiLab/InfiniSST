# HN512 Fast Resume From Step 640 On Aries 8 GPUs

## Hypothesis

Continuing HN512 from the step-640 latest checkpoint on all 8 Aries GPUs should improve wall-clock throughput while preserving the effective 8k global batch. Using `GRAD_CACHE_CHUNK_SIZE=256` should be feasible because the per-rank batch drops to 1024 on 8 GPUs.

## Background / Motivation

The HN512 ablation was paused on Taurus after W&B run `bkcnqlg9` saved a step-640 latest checkpoint. The next run should prioritize speed: use all 8 Aries GPUs, reduce eval overhead by evaluating every 100 steps, and keep a fresh resumable latest checkpoint every 50 train steps.

## What changed vs baseline

- Resume source: `bkcnqlg9` step-640 latest checkpoint.
- Compute: Aries GPUs `0,1,2,3,4,5,6,7`.
- Batch: `BATCH_SIZE=8192`, `PER_GPU_BATCH=1024`, 8 ranks.
- GradCache: `GRAD_CACHE_CHUNK_SIZE=256`.
- Eval cadence: `EVAL_STEPS_SAMPLE=100`.
- Latest checkpoint cadence: `SAVE_LATEST_STEPS=50`, overwriting `_latest.pt`.
- Hard negatives: `HARD_NEG_K_PER_SAMPLE=512`, `HARD_NEG_K=0`.
- Scheduler and best trackers are preserved on resume.
- ACL remains readout-only and must not be used for tau, checkpoint, hyperparameter, or ablation winner selection.

## Expected metrics

The main success criterion is fast, stable continuation without an OOM. Because eval now happens every 100 steps, the next domain readout after step 640 should occur at step 700. The periodic latest checkpoint should refresh at steps 650, 700, 750, and so on even if best metrics do not improve.

## Verdict

FAILED. The detached launcher later reached torchrun initialization, but the recorded process received SIGTERM at 2026-05-25 00:38:01 UTC before a W&B run id or new latest checkpoint was recorded. The replacement event is `20260526T0001__retriever_train__varctx_lmlb_v3_hn512_gc256_resume640_aries8_fast_latest50`.
