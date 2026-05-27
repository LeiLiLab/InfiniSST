# HN depth smoke — `k=8192`, `TCM off`, `smallest + dense`, 8 GPU

Staged feasibility probe for an extremely deep per-sample HN budget. This is
not part of the initial submitted pair; it is prepared now so that we can
quickly test `k=8192` only if `k=4096` remains healthy.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k8192_tcmoff_smallest_dense_normAGGR_8gpu_smoke`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k8192_tcmoff_smoke_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

If the deeper HN regime is still numerically stable at `k=4096`, then a short
`k=8192` smoke run may still be feasible on Aries 8 GPU. The main goal is to
measure feasibility and step-time, not to claim a new model winner.

## Background / Motivation

The user wants `8192` explored only if the deeper regime keeps working. Because
this setting is far more expensive than the main scout pair, we stage it as a
short smoke test rather than spending budget up front.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **Diff**:
  - hparam `tcm_loss_weight`: `1.0` -> `0.0`
  - hparam `tcm_pos_loss_weight`: legacy shared branch -> `0.0`
  - hparam `tcm_neg_loss_weight`: legacy shared branch -> `0.0`
  - hparam `hard_neg_k_per_sample`: `512` -> `8192`
  - hparam `num_gpus`: `6` -> `8`
  - hparam `per_gpu_batch`: `2048` -> `1536` (global `batch_size=12288` preserved)
  - hparam `grad_cache_chunk_size`: `256` -> `64`
  - hparam `epochs`: historical `3` -> smoke `1`
  - hparam `max_steps`: full run -> `40`
  - hparam `max_train_seconds`: full run -> `5400`
  - data / recipe: unchanged `smallest + dense + normAGGR`
  - code: use the new HN-depth common launcher so only HN depth moves semantically

## Expected metrics

- no OOM through step `40`
- `train/step_time_ms <= 120000`
- no obvious collapse in the first eval windows
- if any of the above fail, retire `k=8192` without promoting it

## Verdict

<!-- Filled only if this staged smoke is submitted later. -->
