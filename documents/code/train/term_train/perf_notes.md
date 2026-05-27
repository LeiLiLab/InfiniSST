# qwen3_glossary_neg_train perf profiling notes

## Setup

- Taurus, 2x A6000, world_size=2, per-GPU batch=2048, grad_cache chunk=512.
- Hyperparams cloned from 43849 (`q3_ablA_k1024_norm`).
- `torch.profiler` schedule: wait=2, warmup=3, active=3, repeat=1 → 3 active steps.

## Run log

### Smoke 43856 (first attempt, not annotated)

- 10 active steps, no record_function annotations.
- Trace 4.85 GB (too big for easy analysis); key_averages.txt empty due to
  `key_averages()` being called AFTER `profiler.stop()`.
- Headline: GPU kernel self-time / wall ≈ 69% → ~31% "idle" from the GPU side.
- Top GPU kernels: flash-attn fwd+bwd ~31%, gemm ~21%, elementwise ~27%,
  layernorm 2%, avg_pool1d (MaxSim) 0.07%.
- Top CPU ops: cudaLaunchKernel 88 s, aten::copy_ 76 s, aten::index 55 s,
  aten::nonzero 20 s, cudaStreamSynchronize 28 s.

### Smoke 43863 (annotated, active=3)

- Trace 1.45 GB. key_averages.txt 33 KB, populated correctly.
- Added `record_function`: `gc/prep_h2d`, `gc/phase1_no_grad_fwd`,
  `gc/p1_retriever_fwd`, `gc/p1_text_fwd`, `gc/phase2_loss_plus_emb_bwd`,
  `gc/phase2_bwd_to_embs`, `gc/phase3_refwd_bwd_to_weights`,
  `gc/p3_retriever_fwd`, `gc/p3_text_fwd`, `gc/p3_bwd`, `gc/optimizer_step`.

#### CUDA self-time top contributors (3 active steps, total 58.8 s)

| op | CUDA self | % | calls |
|---|---|---|---|
| `DistributedDataParallel.forward` | 25.491 s | 43.3% | 48 (12 per step per phase) |
| `gc/phase2_loss_plus_emb_bwd` | 24.834 s | 42.2% | 3 |
| `aten::copy_` | 12.294 s | 20.9% | 152,784 |
| `Command Buffer Full` (runtime) | 9.385 s | 16.0% | 75,335 |
| flash attention fwd kernel | 8.715 s | 14.8% | 1152 |
| `aten::mm` | 8.385 s | 14.3% | 41,229 |
| `aten::addmm` | 5.661 s | 9.6% | 10,440 |
| Memcpy HtoD (Pageable→Device) | 5.114 s | 8.7% | 141 (avg 36.3 ms each) |
| `gc/phase2_bwd_to_embs` | **62.7 ms** | 0.11% | 3 |
| `gc/optimizer_step` | 64.0 ms | 0.11% | 3 |

#### CPU self-time top consumers

| op | CPU self | % | calls | CPU avg |
|---|---|---|---|---|
| `Command Buffer Full` | 27.358 s | 26.4% | 75,335 | 363 µs |
| `cudaLaunchKernel` | 17.543 s | 16.9% | 285,186 | 93 µs |
| `cudaMemcpyAsync` | 15.830 s | 15.3% | 119,436 | 133 µs |
| `aten::index` | 13.045 s | 12.6% | **30** | **505 ms** |
| `cudaStreamSynchronize` | 8.337 s | 8.0% | 61,764 | 135 µs |
| `gc/p3_bwd` | 8.246 s | 8.0% | 12 | 2.2 s |
| `aten::_local_scalar_dense` | 144 ms self (2.24 s total) | — | **65,592** | 34 µs |
| `aten::item` | 68 ms self (2.31 s total) | — | **65,592** | 35 µs |

## Interpretation

### The bottleneck is the host, not the GPU

- GPU kernel time across 3 active steps: 58.8 s = 19.6 s/step.
- CPU wall time across 3 active steps: ~103 s = 34 s/step.
- → GPU utilisation ≈ 58%; ~42% of each step the GPU is starved.

### Root causes of starvation

1. **`aten::_local_scalar_dense` = 65,592 calls in 3 steps ≈ 21,900 `.item()` per step.**
   Each forces `cudaStreamSynchronize`. This explains 62 k stream syncs and is
   the dominant "stop the GPU, wait for one scalar" overhead. The 3 sites we
   originally suspected (`_maxsim_score_mfa` fallback at L1800/L1961, eval
   nonzero at L2482/L2965) are only a handful per step. The other ~21.8 k
   must come from inside `Qwen3OmniMoeAudioEncoder` / DDP / collate
   (feature_lens.tolist, MoE router, pad_sequence dynamic shapes, etc.).
2. **`Command Buffer Full` = 27 s on the CPU (26% of wall).**
   CUDA work-queue saturation. GPU can't keep up with the flood of
   sub-100 µs kernels (cudaLaunchKernel avg 93 µs, 285 k launches ≈ 95 k/step).
   Mitigation = fewer launches (torch.compile, CUDA graphs) or larger fused
   kernels.
3. **`aten::index` = 30 calls @ 505 ms CPU avg = 15 s.**
   Only 4 ms of actual GPU work per call (10 ms CUDA total / 30). So this is
   a host-side index-construction path (likely a large Python list → tensor
   conversion, or fancy-indexing going through the slow path).
4. **`Memcpy HtoD Pageable→Device` = 141 calls, 36.3 ms avg, 5.1 s total.**
   Classic "forgot `non_blocking=True` + unpinned" symptom. Targets:
   the `.to(device)` ladder at L1365-L1474 in `gradcache_train_step` and
   its clone at L4443-L4533 in the fallback path, plus the
   `_build_window_time_ranges` + `.to(device)` pattern at L1468-L1469.

### What the *original* hypothesis got wrong

- "`_multiscale_pool` avg_pool1d is the hot spot" — no: 0.07% of GPU time.
  `cumsum` rewrite would buy nothing. Killing that line of work.
- "GPU is idle for well-defined reasons" — partly. 58% util number is right,
  but the dominant cost is **CPU saturation** (flood of tiny kernel launches
  + memcpys + `.item()` syncs), not GPU kernel density.

## Fix priority (revised after 43863)

1. **Syncs**: kill the known `.item()` sites (L1800, L1961, L2482 in eval,
   L2965 eval). Low gain individually (~handful of calls each) but keeps
   future profiles clean. The fat 21.8 k `.item()` calls per step likely
   come from inside `Qwen3OmniMoeAudioEncoder.forward` and
   `transformers` internals — can only be attacked via `torch.compile`
   or by patching upstream, both heavy.
2. **`non_blocking=True` + pinned batch**: every `.to(device)` in
   `gradcache_train_step` and its clone. Batch already pin-memory in
   DataLoader (L3740/3754/3778) so the H2D copies *can* overlap compute
   once `non_blocking=True` is passed.
3. **Cache `_build_window_time_ranges`** on GPU keyed on `(windows, stride,
   T_enc)` so the per-step `torch.tensor([...]).to(device)` pattern collapses
   to a cached GPU tensor reference.
4. **Move `mfa_term_starts`/`mfa_term_ends` into the pinned collate dict**
   so the H2D transfer overlaps with the retriever forward.

Lower-ROI / deferred:

- `torch.compile` the retriever forward (kills launch overhead + MoE `.item()`
  bombs). Big risk, requires static shapes → would need padded-T feature
  extraction first.
- CUDA graphs for the stable path.
