# HN512 Allocation-Compatible Resume From Step 640 On Aries 6 GPUs

## Hypothesis

The HN512 latest checkpoint from `bkcnqlg9` is undertrained for Figure 5, so continuing it on the currently Slurm-available Aries GPUs should produce a cleaner HN512 ablation point without waiting for all 8 GPUs.

## Background / Motivation

The dev-only fixed-raw Figure 5 eligibility check on W&B run `iz1x2v3o` showed that the step-640 HN512 latest checkpoint can be evaluated, but its precision/drop curve sits below HN256/HN1024 in the figure window. The 8-GPU Slurm job `45309` could not start because two Aries GPUs are allocated to another user's idle shell jobs, even though `nvidia-smi` shows no CUDA processes. This event replaces the pending 8-GPU job with a 6-GPU allocation-compatible resume that stays inside Slurm allocation boundaries.

Resume checkpoint:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt`

## What changed vs baseline

- Resume source: `bkcnqlg9` step-640 latest checkpoint.
- Compute: Aries partition, 6 Slurm-allocated GPUs.
- Batch: `BATCH_SIZE=8190`, `PER_GPU_BATCH=1365`, 6 ranks.
- GradCache: `GRAD_CACHE_CHUNK_SIZE=128`.
- Eval cadence: `EVAL_STEPS_SAMPLE=100`.
- Latest checkpoint cadence: `SAVE_LATEST_STEPS=50`, overwriting `_latest.pt`.
- GPU selection: no hard-coded physical GPU list; common preflight uses `SLURM_JOB_GPUS` / `CUDA_VISIBLE_DEVICES`.
- Storage: logs, checkpoints, and W&B files go under `/mnt/gemini/home/jiaxuanluo`.
- Scheduler and best trackers are preserved on resume.
- ACL remains readout-only and must not select tau, checkpoint, hyperparameters, or ablation winner.

## Expected metrics

The immediate success criterion is startup verification: load step 640, initialize a new W&B run, and reach training without OOM. The next useful checkpoint evidence is a refreshed latest checkpoint at step 650 and the next dev/ACL/medicine eval after step 700.

## Verdict

STARTUP VERIFIED. Slurm job `45311` started on Aries with GPU allocation `2,3,4,5,6,7`, loaded the HN512 latest checkpoint at epoch 4 step 640, restored `best_metric_value=0.9881` for `eval_dev/recall@10_gs10000` and `best_metric_secondary_value=0.9889` for `eval_acl6060/recall@10`, and initialized W&B run `mwwe1l1e`.

CANCELLED_USER_REQUESTED at 2026-05-26T02:56:51Z. The user requested terminating this run because HN512 is no longer needed. `scancel 45311` removed the Slurm job; after cleanup, PIDs `3965249`-`3965254` and torchrun PID `3965169` were absent, and GPUs `2,3,4,5,6,7` had no `pmon` processes with only 2-5 MiB residual memory.

W&B:

`https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/mwwe1l1e`

Training setup after resume: `train=7,228,866`, `dev=12,564`, `acl_dev=3,852`, `medicine_dev=11,071`, `world_size=6`, `per_rank_bs=1365`, `total_steps=5292`.
