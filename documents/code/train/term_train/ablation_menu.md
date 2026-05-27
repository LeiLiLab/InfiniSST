# Ablation Menu (post-43848 back-fill)

Ranked by expected impact per GPU-hour on `eval_acl6060/recall@10_gs10000` + `noterm_noise@tau_0p80_gs10000`, anchored on 43848 / `zv28ve3q` (`best/metric_value=0.9395`, `best_secondary/metric_value=0.8744`, `noise=1.47`, tcm viol pos/neg=0.232/0.125).

Scheduled ablations on the Aries 6-GPU slot:

- **A1** (DONE — `tys70s0y`, early stop): `hnps_k512_smallest_dense_normAGGR_6gpu` — per-sample HN k=512. Confirmed per-sample HN > pool HN at matched steps (+0.026 gs10000, +0.011 filt_R@0.80 at step 520).
- **A2** (RETIRED — `we6ssrn7`, killed @ step 260): `hnps_k512_tcm5x_relaxed_smallest_dense_normAGGR_8gpu` — per-sample HN k=512 (stacked from A1) + TCM weight 5x + beta=0.70 / alpha=0.40 + grad_cache_chunk=512 + 8 GPUs. Unattributable: α/β/λ 三变量同改,加 warmup,四变量齐动 → 无法归因失败根因。Post-mortem diagnosis (see run verdict + histogram snapshots 43899/43900): `grad_cache_chunk_size` 和 GPU 数对 in-batch neg 池构造零影响 ([`qwen3_glossary_neg_train.py:2124`](qwen3_glossary_neg_train.py) all-gather),真实变量只剩 α/β/λ,仍然三合一。放宽 α=0.40 把 per-sample HN 挖出来的硬 neg(稳态贡献 20% viol_rate)从 TCM 视野里撤走,反而让 noise@tau0.80_gs10k 从 tys70s0y 1.57 炸到 5.35。
- **C1** (RETIRED — SLURM `43901`, WandB `ll5a6p9k`, status flipped to `failed`): `hnps_k512_lambda5_smallest_dense_normAGGR_6gpu` — 从 `tys70s0y` 基线**只**抬 `tcm_loss_weight 1→5` 的单变量实验已结束，用户判定该方向 low ROI，计算资源转移到 TCM-v2。Launcher: [`run_mfa_smallest_dense_lambda5_6gpu_aries.sh`](run_mfa_smallest_dense_lambda5_6gpu_aries.sh)。Notes: [`notes_lambda5_only.md`](notes_lambda5_only.md)。
- **Phase 0 lock** (DONE): dense ACL6060 `gs10000` tau sweep on the frozen `tys70s0y` audit dump selected `tau*=0.80`; therefore the first TCM-v2 neg threshold is fixed at `0.78`. Record: [`documents/code/offline_evaluation/tys70s0y_dense_tau_lock.md`](../../offline_evaluation/tys70s0y_dense_tau_lock.md).
- **E1** (ARCHIVED — SLURM `43912` cancelled, WandB `ailr03qx` flipped to `status:failed`): `hnps_k512_tcmv2_topk32_smallest_dense_normAGGR_6gpu` — stopped at step `140` after the user decided TCM should be deferred until the non-TCM HN-depth sweet spot is fixed. Archive verdict: strategy pivot, no conclusion drawn. Launcher: [`run_mfa_smallest_dense_tcm_v2_topk32_6gpu_aries.sh`](run_mfa_smallest_dense_tcm_v2_topk32_6gpu_aries.sh). Notes: [`notes_tcm_v2_e1_topk.md`](notes_tcm_v2_e1_topk.md).
- **E2** (ARCHIVED BEFORE START — SLURM `43913` cancelled in queue): `hnps_k512_tcmv2_allscope_smallest_dense_normAGGR_6gpu` — never launched because the sweep priority changed from TCM-first to HN-depth-first. Launcher: [`run_mfa_smallest_dense_tcm_v2_allscope_6gpu_aries.sh`](run_mfa_smallest_dense_tcm_v2_allscope_6gpu_aries.sh). Notes: [`notes_tcm_v2_e2_all_control.md`](notes_tcm_v2_e2_all_control.md).
- **TCM-v2 audit chain** (CANCELLED): dependent audit jobs `43914/43915/43916` were cancelled with E1/E2. TCM moves to the last stage after the HN-depth scout freezes the base retriever recipe.
- **NEXT: HN depth scout (TCM off)** — explore `hard_neg_k_per_sample` under the current `smallest + dense + normAGGR` recipe with `tcm_loss_weight=0`, treating `tau=0.80` only as a fixed probe rather than a tuned deployment threshold. Protocol: [`hn_depth_scout_protocol.md`](hn_depth_scout_protocol.md).
- **H1 completed** (SLURM `43929`, WandB `fma3wmh2`, `status:success`): `hnps_k1024_tcmoff_smallest_dense_normAGGR_8gpu_scout` — clean non-TCM scout on the current recipe. At matched step `200` versus the historical `k=512` reference `tys70s0y`, `k=1024` improves `ACL gs10000 recall@10` by `+3.10pp` (`0.7039 -> 0.7349`), `ACL filt@tau0.80_gs10000` by `+2.33pp` (`0.4961 -> 0.5194`), and `DEV filt@tau0.80_gs10000` by `+1.76pp` (`0.7670 -> 0.7846`), with slightly worse ACL noise (`+0.088`). Promote as the current best HN-depth candidate. Launcher: [`run_mfa_smallest_dense_hnps_k1024_tcmoff_8gpu_aries.sh`](run_mfa_smallest_dense_hnps_k1024_tcmoff_8gpu_aries.sh). Notes: [`notes_hnps_k1024_tcmoff_scout.md`](notes_hnps_k1024_tcmoff_scout.md).
- **H1.5 completed** (SLURM `43932`, WandB `6s3jr70q`, `status:success`): `hnps_k2048_tcmoff_smallest_dense_normAGGR_8gpu_scout` — follow-up scout using `fma3wmh2` as the confirmed baseline, keeping `GRAD_CACHE_CHUNK_SIZE=256` fixed to match the successful `k=1024` run. Completed cleanly and becomes the current HN-depth scout winner, with better dense-bank retrieval at the secondary checkpoint and a modest fixed-threshold noise increase to monitor. Launcher: [`run_mfa_smallest_dense_hnps_k2048_tcmoff_8gpu_aries.sh`](run_mfa_smallest_dense_hnps_k2048_tcmoff_8gpu_aries.sh). Notes: [`notes_hnps_k2048_tcmoff_scout.md`](notes_hnps_k2048_tcmoff_scout.md).
- **H1.75 first submit cancelled** (SLURM `43933`): `hnps_k4096_tcmoff_bs6k_smallest_dense_normAGGR_8gpu_scout` entered setup but WandB init failed because the `variant:` tag was 65 characters, exceeding the 64-character limit. Cancelled immediately and relaunched with shortened tag `hnps_k4096_bs6k_tcmoff_sd_8gpu`. This is not a valid experiment run.
- **H1.75 completed** (SLURM `43934`, WandB `iaiyi1m8`, `status:success`): `hnps_k4096_bs6k_tcmoff_sd_8gpu` — retry `k=4096` by halving local batch from `1536` to `768` (`global 12288 -> 6144`) so the dominant `B_local * K` MaxSim memory term is approximately matched to the successful `k=2048` run, while doubling scout length to `400` steps for a rough sample-budget match. Completed cleanly and becomes a sample-budget-matched HN-depth winner over `k=2048`, with the important caveat that fixed-threshold noise increases and still needs calibration. Baseline: `6s3jr70q`. Launcher: [`run_mfa_smallest_dense_hnps_k4096_tcmoff_bs6k_8gpu_aries.sh`](run_mfa_smallest_dense_hnps_k4096_tcmoff_bs6k_8gpu_aries.sh). Notes: [`notes_hnps_k4096_tcmoff_bs6k_scout.md`](notes_hnps_k4096_tcmoff_bs6k_scout.md).
- **H1.9 first submit failed** (SLURM `43944`, no WandB run): `hnps_k8192_bs2304_tcmoff_sd_6gpu` failed during model load because the shared common launcher selected `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` on Taurus despite physical GPUs `3` and `5` already being busy. This was a launcher GPU-selection failure, not evidence against `k=8192`.
- **H1.9 resubmitted** (SLURM `43945`, WandB `9b9tcinm`, running): `hnps_k8192_bs2304_tcmoff_sd_6gpu` — final HN-size scout on Taurus 6 GPU, now with `SELECT_CLEAN_GPUS=true` so the launcher refuses busy GPUs and selects clean physical devices. Uses `hard_neg_k_per_sample=8192`, `PER_GPU_BATCH=384`, global batch `2304`, `GRAD_CACHE_CHUNK_SIZE=128`, and `MAX_STEPS=1080` to roughly match the sample-budget scale of `k2048@200` / `k4096@400` while keeping per-rank `B_local * K` MaxSim memory near the successful `k4096 bs6k` regime. Baselines: `6s3jr70q iaiyi1m8`. Startup passed WandB init and entered the training loop. Launcher: [`run_mfa_smallest_dense_hnps_k8192_tcmoff_bs2304_6gpu_taurus.sh`](run_mfa_smallest_dense_hnps_k8192_tcmoff_bs2304_6gpu_taurus.sh). Notes: [`notes_hnps_k8192_tcmoff_bs2304_taurus_scout.md`](notes_hnps_k8192_tcmoff_bs2304_taurus_scout.md).
- **H2 failed at launch** (SLURM `43930`, WandB `wwlodnqh`): `hnps_k4096_tcmoff_smallest_dense_normAGGR_8gpu_scout` — OOM at step `1` inside per-sample MaxSim (`torch.einsum("bwd,bkd->bwk", speech_embs, hn_embs)`). With `8 x 1536`, dense windows, and `k=4096`, each rank attempted an extra `12 GiB` allocation on top of ~`41 GiB` already resident. Conclusion: `4096` is infeasible under the current batch/recipe without further memory reduction. Launcher: [`run_mfa_smallest_dense_hnps_k4096_tcmoff_8gpu_aries.sh`](run_mfa_smallest_dense_hnps_k4096_tcmoff_8gpu_aries.sh). Notes: [`notes_hnps_k4096_tcmoff_scout.md`](notes_hnps_k4096_tcmoff_scout.md).
- **H2 retry failed** (SLURM `43931`, no valid WandB run): `hnps_k4096_tcmoff_smallest_dense_normAGGR_8gpu_retry_chunk64` — clean retry after the step-1 OOM, lowering only `grad_cache_chunk_size` from `128` to `64`, still failed at step `1` on the same per-sample MaxSim `einsum("bwd,bkd->bwk")` allocation. Conclusion: chunk reduction does not solve the `k=4096` bottleneck under `8 x 1536` and dense windows. Separate issue: WandB init failed because the retry tag exceeded the 64-character limit, so this run must be tracked from SLURM logs only. Launcher: [`run_mfa_smallest_dense_hnps_k4096_tcmoff_retry_chunk64_8gpu_aries.sh`](run_mfa_smallest_dense_hnps_k4096_tcmoff_retry_chunk64_8gpu_aries.sh). Notes: [`notes_hnps_k4096_tcmoff_retry_chunk64.md`](notes_hnps_k4096_tcmoff_retry_chunk64.md).
- **H3 staged only** (not submitted): `hnps_k8192_tcmoff_smallest_dense_normAGGR_8gpu_smoke` — feasibility smoke prepared but intentionally held back until `4096` proves healthy. Launcher: [`run_mfa_smallest_dense_hnps_k8192_tcmoff_smoke_8gpu_aries.sh`](run_mfa_smallest_dense_hnps_k8192_tcmoff_smoke_8gpu_aries.sh). Notes: [`notes_hnps_k8192_tcmoff_smoke.md`](notes_hnps_k8192_tcmoff_smoke.md).

**锁定决定 (post-TCM pivot)**: 先把 TCM 完全后置。当前阶段只比较非-TCM HN 深度，不再用 ACL 结果反推 TCM margin/weight；`tau=0.80` 在 HN scout 中仅作为固定 probe 观测 `filtered_recall/noise`，不作为最终部署阈值结论。

Everything below is deferred; ordering reflects priority for the next planning cycle.

---

## M1 (HIGHEST): On-the-fly audio augmentation

**Why first**: OOD decomposition on `r0xi5xkt` showed -8.4pp of the -10.4pp DEV-to-ACL drop comes from acoustic shift (seen-in-train terms still drop 8.4pp on ACL audio). A1 and A2 attack embedding geometry, not input-distribution diversity. On-the-fly augmentation directly attacks that 8.4pp acoustic gap.

**Touch-points**:
- Dataloader hook in `documents/code/train/term_train/qwen3_glossary_neg_train.py` around the `_load_audio_and_mfa` codepath.
- `audiomentations` or `torchaudio.sox_effects` pipeline: p=0.5 RIR convolution (OpenSLR-28), p=0.3 additive noise (MUSAN 0-20dB SNR), p=0.3 codec (opus@32k / speex), p=0.2 low-shelf EQ. Apply BEFORE MFA window selection so windows align with augmented audio.
- Cache per-epoch or regenerate every N steps (memory vs diversity tradeoff).
- Log `train/aug/rir_applied_rate` etc. as WandB histograms so we can verify the pipeline fires.

**Expected**: +1.0 to +2.5pp on `best_secondary/metric_value`. **Cost**: 1.3-1.5x step time (GPU idle during CPU aug; mitigate `num_workers>=8`).

## M4 (easiest win): Re-eval 43848 best-ckpt on a tau sweep

**Why**: `noise=1.47` at tau=0.80 is just the coarse grid point. Rescanning `best/step=1240` on tau `{0.70, 0.72, ..., 0.90}` for `filtered_recall` and `noterm_noise` likely uncovers tau~0.82 where `noise` drops to ~1.0 and `filtered_recall` barely moves. No retraining.

**Touch-points**:
- Extend `documents/code/offline_evaluation/threshold_sweep_maxsim.py` with the `zv28ve3q` ckpt path and the fine tau grid.
- Emit one WandB run per tau into `simuleval_eval` with `task:eval`, `family:sst_ood_hardneg`, `trained_from_run:zv28ve3q`.
- **Cost**: ~2 CPU-hours; no GPU training.

**Expected**: same `best_secondary/metric_value`; -0.3 to -0.5 on `noise` at new operating tau, +0.5-1pp on `filtered_recall`.

## M3: Online hard-negative mining

**Why**: Current HN set is fixed per-epoch. As the model improves, the hardest negatives shift; offline mining lags. A small online refresh every ~500 steps (running negative bank of current model embeddings) sharpens the boundary late in training.

**Touch-points**:
- Flags `--neg_bank_size`, `--online_hard_neg_k`, `--neg_bank_refresh_steps` already exist in `qwen3_glossary_neg_train.py` (zero in 43848). Try `50000 / 64 / 500`.
- Code may have bit-rotted since variantE baseline; smoke run (`MAX_STEPS=100`) first.

**Expected**: +0.3 to +0.8pp on `best_secondary/metric_value`. **Cost**: +10-15% step time at refresh steps; otherwise neutral.

## M5 (lowest here): MFA-acoustic hard-neg mining

**Why**: Current HN comes from a wiki-disjoint text glossary. True acoustic hard negatives (phonetically / contextually confusable in audio) are never mined. Build via maxsim of the 43848 best-ckpt over the 10k wiki glossary at the per-sample level, pre-compute offline to `wiki_hard_neg_acoustic.json`, merge with the text HN on-the-fly via a new `--hard_neg_acoustic_glossary` flag.

**Expected**: speculative, +0.5 to +1.5pp. **Cost**: 2-3 days offline infra + smoke + full train.

## Dropped: M2 text-side unfreeze

`TEXT_LR="0"` in the launcher does NOT freeze the text encoder: `qwen3_glossary_neg_train.py:3641-3643` falls back to `args.lr` when `text_lr<=0`, so text LoRA already trains at `1.7e-4` in 43848 (same as the acoustic side). There is no frozen text encoder to unfreeze. If we want to ablate text LR, pass an explicit non-zero `TEXT_LR` (e.g. `0.5*args.lr`). Truly freezing would need a new CLI flag plus `requires_grad=False` on the text LoRA params; not worth the plumbing for a likely-negative result.
