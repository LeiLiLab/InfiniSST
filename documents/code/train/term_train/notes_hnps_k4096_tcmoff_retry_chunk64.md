# HN depth retry — `k=4096`, `TCM off`, `smallest + dense`, 8 GPU, `chunk=64`

Clean retry for the `k=4096` HN-depth scout after the first launch
(`wwlodnqh` / SLURM `43930`) OOMed at step `1`. This retry keeps the same
semantic experiment setup and only reduces the grad-cache chunk size from `128`
to `64` to test whether the earlier failure came from chunk-local recompute
pressure rather than from the HN depth being fundamentally impossible.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k4096_tcmoff_smallest_dense_normAGGR_8gpu_retry_chunk64`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hnps_k4096_tcmoff_retry_chunk64_8gpu_aries.sh`
- **Primary baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2

## Hypothesis

If the first `k=4096` failure was driven by chunk-local recompute pressure, then
lowering `grad_cache_chunk_size` from `128` to `64` should allow the run to get
past step `1` and into the first eval window without changing the retrieval
recipe itself.

## Background / Motivation

The first `k=4096` attempt (`wwlodnqh`) OOMed immediately in
`_maxsim_score_mfa_per_sample()` while computing `torch.einsum("bwd,bkd->bwk", ...)`.
Although the retry hypothesis that "the previous `k=1024` job had just exited"
is not strongly supported by the preflight GPU check, a clean retry with a
smaller grad-cache chunk is still worth one controlled test before declaring
`k=4096` infeasible.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
- **Diff**:
  - hparam `hard_neg_k_per_sample`: `1024` -> `4096`
  - hparam `tcm_loss_weight`: `0.0` (unchanged; TCM remains fully off)
  - hparam `tcm_pos_loss_weight`: `0.0` (unchanged)
  - hparam `tcm_neg_loss_weight`: `0.0` (unchanged)
  - hparam `grad_cache_chunk_size`: `256` in the `k=1024` scout, `128` in the first `k=4096` attempt, now `64`
  - hparam `max_steps`: `200` in the `k=1024` scout -> `120`
  - hparam `max_train_seconds`: `14400` in the `k=1024` scout -> `12600`
  - data / recipe / batch size: unchanged `smallest + dense + normAGGR`, `8 x 1536 = 12288`
  - direct predecessor failure reference: `wwlodnqh`

## Expected metrics

- survive past step `1` and into the first eval window
- no OOM in the first `40` steps
- `train/step_time_ms <= 120000`
- if it reaches step `80`, metrics should not obviously collapse relative to the `k=1024` scout

## Verdict

<!-- Filled after the retry completes. -->
