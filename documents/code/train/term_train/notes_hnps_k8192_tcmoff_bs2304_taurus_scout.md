# HN depth scout — `k=8192`, `TCM off`, `smallest + dense`, 6 GPU Taurus

Final HN-size scout after `k=2048` completed cleanly and `k=4096` showed
positive sample-matched filtered-recall signals. This run pushes per-sample HN
to `k=8192` on Taurus 6 GPU while shrinking local batch so the dominant
per-rank MaxSim memory term stays near the successful `k=4096` memory regime.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k8192_bs2304_tcmoff_sd_6gpu`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k8192_tcmoff_bs2304_6gpu_taurus.sh`
- **Baseline runs**:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/6s3jr70q
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8

## Hypothesis

If the HN-depth trend still holds beyond `k=4096`, then replacing more
in-batch/easy negatives with `k=8192` curated per-sample hard negatives should
improve dense-bank filtered recall at a matched sample budget. Because the
global batch is reduced to `2304`, this scout uses `1080` steps to roughly match
the `k=2048@200` / `k=4096@400` sample-budget scale.

## Background / Motivation

The `k=2048` scout (`6s3jr70q`) became the first clean non-TCM HN-depth winner.
The `k=4096` line (`iaiyi1m8`) required halving local batch to avoid MaxSim OOM,
and sample-matched comparison showed meaningful filtered-recall gains while
also increasing fixed-threshold noise. This final scout tests whether the
benefit continues at `k=8192` or whether the noise/memory cost becomes the
dominant failure mode.

## What changed vs baseline

- **Baseline run URLs**:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/6s3jr70q
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iaiyi1m8
- **Diff vs `iaiyi1m8`**:
  - hparam `hard_neg_k_per_sample`: `4096` -> `8192`
  - hparam `num_gpus`: `8` -> `6`
  - hparam `per_gpu_batch`: `768` -> `384`
  - hparam `batch_size`: `6144` -> `2304`
  - hparam `grad_cache_chunk_size`: `256` -> `128`
  - hparam `max_steps`: `400` -> `1080`
  - hparam `max_train_seconds`: `21600` -> `36000`
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - data / recipe: unchanged `smallest + dense + normAGGR`

## Expected metrics

- no OOM in the first `40` steps
- by step `1080`, compare primarily against `k2048@200` and `k4096@400` by sample budget
- `eval_dev/topk10_filtered_recall@tau_0p80_gs10000` should not regress below the `k4096` matched-budget line
- `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000` should improve enough to justify any fixed-threshold noise increase
- fixed-threshold noise is a diagnostic; final selection should consider filtered recall, recall@10, and noise together

## Verdict

<!-- Filled after the scout completes. -->
