# HN depth scout — `k=4096`, `TCM off`, `smallest + dense`, 8 GPU, batch 6k

Follow-up HN-depth scout after the successful `k=2048` run. The previous
`k=4096` attempts failed with the original `8 x 1536 = 12288` batch because the
per-sample MaxSim tensor exceeded GPU memory. This run halves the local batch
to `8 x 768 = 6144` so the dominant `B_local * hard_neg_k_per_sample` memory
term is approximately matched to the successful `k=2048, batch 12288` run.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k4096_bs6k_tcmoff_sd_8gpu`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_bs6k_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/6s3jr70q

## Hypothesis

If deeper per-sample HN is the useful direction, then trading half of the
in-batch negatives for `k=4096` curated per-sample hard negatives should improve
dense-domain retrieval after a matched token/sample budget. Because the global
batch is halved, the scout uses `400` steps as the rough compute/sample-budget
counterpart to the `k=2048` `200`-step run.

## Background / Motivation

`k=2048` (`6s3jr70q`) improved the secondary checkpoint over `k=1024` on DEV
and ACL dense-bank retrieval, but also increased fixed-threshold noise. That
pattern suggests HN depth is moving the ranking in the right direction, while
absolute-threshold calibration may need more training or a later TCM stage. The
original `k=4096` run and the chunk-64 retry both failed at step `1`, so this
run changes memory pressure by reducing `PER_GPU_BATCH` instead of only changing
gradient-cache chunking.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/6s3jr70q
- **Diff**:
  - hparam `hard_neg_k_per_sample`: `2048` -> `4096`
  - hparam `per_gpu_batch`: `1536` -> `768`
  - hparam `batch_size`: `12288` -> `6144`
  - hparam `max_steps`: `200` -> `400`
  - hparam `max_train_seconds`: `18000` -> `21600`
  - hparam `grad_cache_chunk_size`: `256` -> `256` (kept fixed initially)
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - data / recipe: unchanged `smallest + dense + normAGGR`

## Expected metrics

- no OOM in the first `40` steps
- by step `400`, `eval_dev/topk10_filtered_recall@tau_0p80_gs10000 >= 0.80`
- by step `400`, `eval_acl6060/recall@10_gs10000 >= 0.75`
- by step `400`, `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000 >= 0.56`
- fixed-threshold noise should be read as a diagnostic, not the primary scout selector

## Verdict

SUCCESS (scout): `k=4096` with half local batch completed cleanly and is a
sample-budget-matched HN-depth winner over the completed `k=2048` scout on the
main dense-bank filtered-recall probes. The main trade-off is a clear increase
in fixed-threshold noise, so this should be treated as a stronger retrieval
candidate that still needs threshold/noise calibration before deployment.
