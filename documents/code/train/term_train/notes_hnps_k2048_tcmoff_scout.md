# HN depth scout — `k=2048`, `TCM off`, `smallest + dense`, 8 GPU

Next HN-depth scout after the successful `k=1024` non-TCM run. This run keeps
the same `smallest + dense + normAGGR` retriever recipe, restores
`GRAD_CACHE_CHUNK_SIZE=256` to match the successful `k=1024` setting, and tests
whether pushing per-sample hard negatives to `k=2048` yields another useful
gain before the regime becomes memory- or noise-limited.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k2048_tcmoff_smallest_dense_normAGGR_8gpu_scout`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k2048_tcmoff_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2

## Hypothesis

If the HN sweet spot is above `k=1024`, then increasing to `k=2048` while
keeping the successful `chunk=256` setting should improve early `gs10000`
retrieval quality on the current recipe. If the regime is already near the
capacity edge, then `k=2048` should either fail early or produce only marginal
gains over `k=1024`.

## Background / Motivation

The `k=1024` scout (`fma3wmh2`) completed cleanly and beat the historical
`k=512` reference at matched step `200`, so the next natural question is
whether that gain continues at `k=2048`. The direct `k=4096` attempt and its
clean retry both failed at step `1`, which suggests the `4096` regime is beyond
the current memory budget. `k=2048` is therefore the next reasonable point to
test with the same successful chunk size used by `fma3wmh2`.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
- **Diff**:
  - hparam `hard_neg_k_per_sample`: `1024` -> `2048`
  - hparam `grad_cache_chunk_size`: `256` -> `256` (kept fixed to match the successful baseline)
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - hparam `tcm_pos_loss_weight`: `0.0` (unchanged)
  - hparam `tcm_neg_loss_weight`: `0.0` (unchanged)
  - hparam `max_steps`: `200` -> `200`
  - hparam `max_train_seconds`: `14400` -> `18000`
  - data / recipe / batch size: unchanged `smallest + dense + normAGGR`, `8 x 1536 = 12288`

## Expected metrics

- no OOM in the first `40` steps
- by step `200`, `eval_dev/topk10_filtered_recall@tau_0p80_gs10000 >= 0.79`
- by step `200`, `eval_acl6060/recall@10_gs10000 >= 0.74`
- by step `200`, `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000 >= 0.52`
- `train/step_time_ms` should remain within a tolerable scout budget for a single-node Aries run

## Verdict

SUCCESS (scout): `k=2048` completed cleanly and improves the secondary
checkpoint over the confirmed `k=1024` baseline on both DEV and ACL dense-bank
retrieval probes, with a modest noise increase at the fixed threshold probe.
Promote `k=2048` as the current HN-depth scout winner, while treating the noise
increase as the main risk to monitor before any longer run.
