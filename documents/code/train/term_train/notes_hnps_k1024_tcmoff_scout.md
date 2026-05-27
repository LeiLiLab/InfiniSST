# HN depth scout — `k=1024`, `TCM off`, `smallest + dense`, 8 GPU

Primary HN-depth scout after the TCM pivot. This run keeps the current
`smallest + dense + normAGGR` retriever recipe, turns TCM fully off, and tests
whether per-sample hard negatives at `k=1024` outperform the historical
`k=512` reference without introducing a second moving axis.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k1024_tcmoff_smallest_dense_normAGGR_8gpu_scout`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k1024_tcmoff_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

With TCM fully disabled, increasing per-sample HN depth from the historical
`k=512` reference to `k=1024` should improve early `gs10000` retrieval quality
under the current `smallest + dense + normAGGR` recipe. The gain should show up
before one full epoch and should not require more than a moderate step-time
increase on Aries 8 GPU.

## Background / Motivation

`tys70s0y` already showed that per-sample HN beats the shared pool path at
matched steps, but that run still carried legacy `TCM=1` and `squared_hinge`.
We are deferring TCM until the last stage, so the first clean question is
whether deeper HN alone helps on the current recipe. Historical anchor
`r0xi5xkt` suggests `k=1024` can work, but it used the older MaxSim window
recipe and is therefore not a clean answer for the current line.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **Diff**:
  - hparam `tcm_loss_weight`: `1.0` -> `0.0`
  - hparam `tcm_pos_loss_weight`: legacy shared branch -> `0.0`
  - hparam `tcm_neg_loss_weight`: legacy shared branch -> `0.0`
  - hparam `hard_neg_k_per_sample`: `512` -> `1024`
  - hparam `num_gpus`: `6` -> `8`
  - hparam `per_gpu_batch`: `2048` -> `1536` (global `batch_size=12288` preserved)
  - hparam `epochs`: historical `3` -> scout `1`
  - hparam `max_steps`: full run -> `200`
  - hparam `max_train_seconds`: full run -> `14400`
  - data / recipe: unchanged `smallest + dense + normAGGR`
  - code: use the new HN-depth common launcher so only HN depth moves semantically

## Expected metrics

- by step `200`, `eval_dev/topk10_filtered_recall@tau_0p80_gs10000 >= 0.76`
- by step `200`, `eval_acl6060/recall@10_gs10000 >= 0.80`
- by step `200`, `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000 >= 0.76`
- `train/step_time_ms <= 55000`
- no OOM and no obvious collapse (`eval_acl6060/recall@10_gs10000` should stay well above the failed-TCM regime)

## Verdict

SUCCESS (scout): completed `200` steps in `2.25h` with no instability. Against
the historical `k=512` reference at matched step `200`, `k=1024` improves
`eval_acl6060/recall@10_gs10000` from `0.7039` to `0.7349` (+3.10pp) and
`eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000` from `0.4961` to
`0.5194` (+2.33pp), while `eval_dev/topk10_filtered_recall@tau_0p80_gs10000`
also rises from `0.7670` to `0.7846` (+1.76pp). The trade-off is slightly worse
ACL noise at the same probe (`1.3496` -> `1.4375`, +0.088). Promote `k=1024`
as the current best HN-depth candidate and keep TCM off for now.
