# Perf A/B bench: qwen3_glossary_neg_train (after nonblock+cache+sync-removal)

## Hypothesis

Removing host-device syncs in `_maxsim_score_mfa` fallback (F.one_hot instead
of `.item()`), plus `non_blocking=True` on every per-step `.to(device)` call,
plus caching `_build_window_time_ranges` tensors on device by
`(windows, stride, T_enc)`, will reduce per-step wall time by >=10% without
changing the loss curve within step-to-step noise over the first 200 steps.

The second annotated profile (job 43863) showed:
  - 31% GPU idle was CPU-saturation, not compute headroom.
  - `cudaStreamSynchronize` ~28s / 10 active steps.
  - Main culprits: `aten::_local_scalar_dense` (`.item()`, 21.9k/step),
    H2D `Memcpy Pageable->Device` (141 calls x 36.3ms avg), and
    `aten::index` at 505ms CPU avg from `_build_window_time_ranges` re-run
    per step.

## Background / Motivation

Baseline profile trace: `/mnt/gemini/home/jiaxuanluo/perf_traces/q3_maxsim_20260423-*`
(jobs 43856 + 43863). Notes: `documents/code/train/term_train/perf_notes.md`.
Original TODO acceptance: step_time_ms reduction >=10% without harming loss
curve; see plan `perf_syncs_+_a1_parallel_91e075b3.plan.md` section A4.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/unite-llm/qwen3_rag (run 43849 `q3_ablA_k1024_norm`, 6 GPUs, same hparams; also compared against the 2-GPU profile job 43863 step-wall-time).
- **Diff**:
  - code: removed `int(win_dur.argmax().item())` sync in `_maxsim_score_mfa` fallback (two sites, L1800+L1968) → pure-GPU `F.one_hot` + bool OR.
  - code: added `non_blocking=True` to every per-step `.to(device)` in `gradcache_train_step` + non-gradcache fallback + `compute_masked_contrastive_loss` + `_maxsim_score_mfa`.
  - code: added module-level cache `_WINDOW_RANGES_CACHE` keyed by `(windows, stride, T_enc, device, dtype)`; `_build_window_time_ranges` results now live on GPU and are reused across steps.
  - code: added `train/step_time_ms` WandB metric so A/B deltas are visible in dashboards.
  - hparam: no change vs 43849 config (same LR, temperature, wiki_rank, maxsim_windows, hard_neg_k_per_sample, grad_cache_chunk_size).
  - GPUs: 2 instead of 6 (scheduling + cost; per-GPU batch unchanged, so per-GPU wall is directly comparable modulo all-reduce time, which is cheaper at 2 GPUs than at 6).
  - max_steps: 200 (bench only, no eval, no save).
  - data: same `term_train_3variant_1m_mfa.jsonl`.

## Expected metrics

- `train/step_time_ms`: 43849 tqdm steady-state ≈ 25.5 s/it on 6 GPUs → 2-GPU pre-fix estimate ≤ 25 s/it. After-fix expected ≤ 22.5 s/it (>=10% drop).
- `train/loss`: curve across steps 0-200 should be within 1-2% of 43849's loss curve at matching global_step (allowing for 6→2 GPU batch differences; same per-GPU batch → same per-GPU loss dynamics).

## Verdict

FAILED 10% acceptance threshold. 2-GPU after-fix steady-state step_time
≈ 24.7 s/it (range 24.4-25.2 across steps 14-74, excluding eval/neg-bank
spike cycles). 43849 6-GPU steady-state was 25.5 s/it; 2-GPU pre-fix
would be ≈24.5-25 s/it (cheaper allreduce at 2 peers). Observed delta
≈1% — within step-to-step noise. Loss curve matches expectation
(step 20 loss=8.39, step 40=8.08, step 60=8.08 — same exponential decay
signature as 43849's opening steps).

Root cause: dominant CPU bottleneck is the 21.9k `.item()` / per step
flood inside `Qwen3OmniMoeAudioEncoder`, confirmed in job 43863's
annotated profile. Our sync-removal + non_blocking + window-cache
targeted known sites in our training code, which account for <5% of
CPU-side sync cost. Upstream encoder internals would require
`torch.compile` or encoder-internal refactor — explicitly out of scope
per the plan's §A2 ("remaining .item() flood is inside
Qwen3OmniMoeAudioEncoder (needs torch.compile, out of scope)").

Action: keep the fixes (no regression, marginal overlap benefit on
multi-GPU allreduce, cleaner code, and window-cache saves repeated
`torch.tensor(...).to(device)` on every step). Do NOT block A1
voice-pool training on further perf work. Next perf attempt should be
`torch.compile` on the retriever forward or an encoder-internal
`.item()` audit.

Bench WandB run: `retriever_perf__after_nonblock_cache_sync__20260423-0450__2gpu_200steps` (id `6mtrqtbv`).
Comparison baseline: 43849 (no WandB step_time_ms metric logged; timing
pulled from tqdm log `25.5 s/it` steady-state after step 200).

