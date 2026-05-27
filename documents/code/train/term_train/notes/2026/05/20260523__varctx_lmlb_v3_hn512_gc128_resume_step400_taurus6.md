# HN512 gc128 resume from step 400 latest

## Hypothesis

Lowering `GRAD_CACHE_CHUNK_SIZE` from 256 to 128 should remove the rank-1 OOM
seen in `gasqw118` while preserving the same HN512 ablation recipe and effective
global batch.

## Background / Motivation

The scheduled HN512 resume `gasqw118` started from the HN512 latest checkpoint,
completed the step-400 eval, saved a latest checkpoint, then failed with CUDA
OOM during the next training step. Taurus GPUs are free again, so this event
continues from that step-400 latest checkpoint with a smaller GradCache chunk.

## What changed vs baseline

- Resume source chain: `5fwrs7rh` -> `gasqw118`.
- Resume checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc256_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_latest.pt`.
- `GRAD_CACHE_CHUNK_SIZE=128` instead of 256.
- Keep `HARD_NEG_K_PER_SAMPLE=512`, `BATCH_SIZE=8190`, `PER_GPU_BATCH=1365`,
  `CUDA_DEVICE_LIST=0,1,2,3,4,5`, TCM off, `SAVE_LATEST_ON_EVAL=true`.
- Preserve best trackers on resume; do not reset scheduler or best metrics.

## Expected metrics

This is an operational resume. The first important success condition is that the
run survives past the post-eval step that failed in `gasqw118`. Later metrics
should be interpreted as the HN512 line, with dev as the primary selection
surface and ACL/medicine as readouts.

## Verdict

PAUSED at user request to free Taurus GPUs. W&B run `yp0rmgrl` reached the
step-560 eval and saved the latest checkpoint before the process was stopped.
Resume from:
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8190_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8190_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn512_tcmoff_ep6_v3_smallest_dense_normAGGR_gpu012345_taurus_latest.pt`.
