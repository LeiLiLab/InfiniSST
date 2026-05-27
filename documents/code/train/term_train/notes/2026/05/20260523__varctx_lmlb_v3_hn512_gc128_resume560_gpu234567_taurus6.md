# HN512 gc128 Resume From Step 560 On Taurus GPUs 2-7

## Hypothesis

Continuing the HN512 ablation from the step-560 latest checkpoint should let the run finish the late-stage comparison without changing the data recipe, hard-negative depth, or checkpoint-selection semantics. The reduced GradCache chunk size of 128 is kept because the prior gc256 resume OOMed, while gc128 reached step 560.

## Background / Motivation

The current HN512 line descends from the lh1b88kw recipe and is part of the hard-negative depth ablation against no-HN, HN256, and HN1024. The previous Taurus gc128 resume run `yp0rmgrl` was intentionally paused after saving the step-560 latest checkpoint to free GPUs.

## What changed vs baseline

- Resume source: `yp0rmgrl` step-560 latest checkpoint.
- GPU placement: Taurus GPUs `2,3,4,5,6,7`.
- Hard negatives: `HARD_NEG_K_PER_SAMPLE=512`, `HARD_NEG_K=0`.
- GradCache: `GRAD_CACHE_CHUNK_SIZE=128`.
- Batch semantics: 6 ranks, `PER_GPU_BATCH=1365`, effective global batch `8190`.
- Scheduler and best trackers are preserved on resume.
- ACL remains readout-only and must not be used for tau, checkpoint, hyperparameter, or winner selection.

## Expected metrics

The main expectation is continuity from step 560 without an OOM or metric reset. The primary checkpoint metric remains `eval_dev/recall@10_gs10000`; secondary remains `eval_acl6060/recall@10`. `SAVE_LATEST_ON_EVAL=true` should keep a resumable latest checkpoint after each evaluation even if best metrics do not improve.

## Verdict

PAUSED at user request to free Taurus GPUs. W&B run `bkcnqlg9` saved a step-640 latest checkpoint before termination:

`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu234567_taurus_latest.pt`

The process was stopped with SIGTERM after that checkpoint was verified. Taurus GPUs `2,3,4,5,6,7` were released.
