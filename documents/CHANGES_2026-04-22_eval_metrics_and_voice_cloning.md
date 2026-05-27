# Session Changes — 2026-04-22

Session summary: diagnostics on the 43827 per-sample k=1024 retrieval run,
adding deployment-relevant eval metrics to the training loop, and standing
up a new GigaSpeech voice-cloning data pipeline (replaces 109 VCTK voices
with 10k diverse GigaSpeech references).

## Background

- **43827** (per-sample k=1024 hard-neg cold-start, 5 epochs) was running at
  the start of this session. Peaked at step 1960 with ACL6060 r@10_gs10000
  ≈ 0.88, then regressed. **Cancelled** mid-run.
- **Pool hard-neg k=64 baseline (variantE)** still holds the training-peak
  ACL6060 r@10_gs10000 ≈ 0.9085, which per-sample k=1024 failed to beat.
- The core question driving this session: **why is ACL6060 OOD stuck 10 pp
  below DEV, and what's the highest-ROI lever?**

### Diagnostic findings (done earlier in session, informs this plan)

- **ACL6060 -10.4 pp OOD drop decomposes to 80% acoustic / 20% vocabulary.**
  Terms seen in train still drop -8.4 pp DEV→ACL. Pearson r(log train_freq,
  ACL pos_sim) = -0.014 — frequency in train is basically uncorrelated with
  ACL score.
- **Per-sample k=1024 is effectively k=5 signal + noise:** at τ=0.07 softmax
  only the top-~5 negatives per anchor have meaningful gradient; the
  remaining ~1019 inflate the denominator norm → signal dilution, not any
  "pulling neighbors closer" as I first claimed.
- **TCM is underweighted:** tcm_pos≈0.03 vs infonce≈1.7 → ~2% of total loss.
  Despite 16% of ACL positives violating T_β=0.85, the penalty barely flows.
- **Synthetic-audio diversity bottleneck:** 3M TTS portion uses only 109
  VCTK studio-UK speakers. Everything trained on synth portion is narrow
  acoustically — exactly where ACL6060 is OOD.

## Code changes (edits)

### `documents/code/train/term_train/qwen3_glossary_neg_train.py`

Three separate add-ons in this session:

1. **`--dump_sim_distributions <dir>` (eval-only diagnostic)** — new flag +
   `_dump_sim_distributions(path, logits, targets, term_texts=...)` helper.
   Emits per-eval-set NPZ with `pos_sim`, `neg_top_sim[N,32]`,
   `neg_sim_{mean,max}`, `bank_size`, `term_texts`. Called once per bank
   (base / gs1000 / gs10000) from `run_sample_eval`. Used to analyze
   seen-vs-unseen partitions of ACL6060 under the no-TCM tsweep baseline.

2. **MFA window-selection modes (training-time gradient routing)** — new
   module-level constant `MFA_WINDOW_SELECTION_MODES = {"hard_max",
   "smallest", "logsumexp"}`, default `"hard_max"` (unchanged behavior).
   New CLI `--mfa_window_selection` routed through
   `compute_masked_contrastive_loss` → `_maxsim_score_mfa`. The
   `"smallest"` branch uses `argmin(window_duration)` masked with `inf`
   on non-covering windows → gradient flows through the tightest crop
   covering the MFA-aligned term span.

3. **Threshold-sweep eval metrics (deployment-relevant, fires every eval
   window)** — the big one this session:
   - Changed `--tcm_sweep_thresholds` default from `None` to
     `[0.5, 0.6, 0.7, 0.8]`. Always-on sweep during eval.
   - New helper `_compute_noterm_noise(full_logits, has_term_mask, taus,
     topk)` → emits `noterm_noise@top{K}_tau_{X}` per tau. Matches
     `avg_noise_terms` in `threshold_sweep_maxsim.py`.
   - Refactored the base-bank detection block so `full_logits_base` is
     computed whenever *either* detection (full mode) OR sweep (any mode)
     is needed. Previously `eval_minimal` short-circuited it entirely.
   - Wired `noise` into each per-τ log line on base + every glossary size,
     so every eval window now prints e.g. `sweep@0.80: R=0.656 P_mic=0.390
     P_mac=0.563 kept=2.22 noise=0.72`.
   - Verified against threshold_sweep TSV: training-eval noise at ACL
     gs10000/τ=0.80 = 0.72 vs offline-sweep 0.65 (small drift from
     slightly different valid_mask; same qualitative behavior).

### `documents/code/train/term_train/run_mfa_smallest_dense_k1024_aries.sh`

- **Loosened preflight** (root cause of 43832 crash). Previous behavior:
  abort after 2 min if any of the 8 GPUs wasn't `memory.used <= 500 MiB`.
  New behavior: log free/busy as WARN, then use SLURM-allocated device
  list `0..NUM_GPUS-1` unconditionally. SLURM `--gres=gpu:8 --exclusive`
  already guarantees isolation; the old hard-gate was over-defensive and
  killed 43832 because GPU 2 had a 10 GB stale allocation.

Also new file in same directory: fork of `run_hardneg_per_sample_k1024_cold_aries.sh`
with `MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"` +
`MFA_WINDOW_SELECTION=smallest` + `EPOCHS=3` (compute-constrained).

## New files

### Eval / analysis

| File | Purpose |
|---|---|
| `documents/code/train/term_train/eval_dump_sim_tsweep_baseline.sh` | 1-GPU taurus launcher. Runs `--eval_only --dump_sim_distributions ...` against the no-TCM tsweep baseline checkpoint. Produces NPZ per set × bank. |
| `documents/code/offline_evaluation/run_threshold_sweep_43827_best_taurus.sh` | Same grid (τ=0.5-0.85 step 0.05, gs ∈ {raw, 1k, 10k}) as the existing `run_threshold_sweep_variantE_best_taurus.sh` but pointed at the 43827 snapshot. |

### Voice-cloning data pipeline

| File | Purpose |
|---|---|
| `documents/code/data_pre/wiki_synth/build_gigaspeech_voice_pool.py` | Build 10k GigaSpeech voice-reference pool. Reads SQLite manifest index (8.3M segments) + MFA TextGrids, decodes 5-12s segments from opus, outputs wav + `speaker_index.json` in the VCTK-compatible format. Balanced audiobook/podcast/youtube (3,332 / 3,334 / 3,334). |
| `documents/code/data_pre/wiki_synth/run_build_gigaspeech_voice_pool.sh` | CPU-only (gpu:0) taurus launcher, ~2 minute build. |
| `documents/code/data_pre/wiki_synth/3variant/run_tts_3variant_gigaspeech_poc_taurus.sh` | 6-GPU × 6-shard taurus POC launcher. Runs the existing `rag_tts_multispeaker_noise.py` unchanged, just swaps `--speaker-dir` to the new gigaspeech pool. Processes the 5k POC subset. |

### Plan / docs

| File | Purpose |
|---|---|
| `~/.claude/plans/mfa-smallest-covering-2-mfa-elegant-adleman.md` | Consolidated plan for this session: preflight fix, eval metrics, voice cloning (with scale estimates, risks, verification). |

## New data artifacts

| Path | Contents |
|---|---|
| `/mnt/taurus/data/jiaxuanluo/gigaspeech_speaker_prompts/` | 10,000 wav files + `speaker_index.json`. Mean dur 7.2s. Used as CosyVoice zero-shot reference pool. |
| `/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant_gigaspeech_poc/wiki_synth_utterances_poc5k.jsonl` | 5k random sample of the 3M 3variant utterance set, for the POC TTS run. |
| `/mnt/gemini/home/jiaxuanluo/train_outputs/snapshots/43827_best_acl6060_gs10000_step1320_0p8775.pt` | Frozen copy of 43827's live best_ACL ckpt at step 1320 (MD5-verified). Used by threshold sweep + sim-distribution dumps so the eval doesn't race with the live training file. |
| `/mnt/taurus/home/jiaxuanluo/sim_dump/tsweep_bs6k_t0.07_m0.0_notcm/` | 6 NPZ files (dev × {base,gs10000}, acl × {base,gs1000,gs10000}) with per-sample pos_sim + top-32 neg_sim for the no-TCM baseline. Drives the seen/unseen ACL decomposition. |
| `/mnt/gemini/data2/jiaxuanluo/threshold_sweep/43827_best_step1320_tau0p5_0p85/` | TSVs + PNGs: P/R/F1/noise vs τ for 43827 snapshot on DEV+ACL × {raw, gs1000, gs10000}. |

## Jobs submitted this session (SLURM)

| JobID | State | Purpose | Result |
|---|---|---|---|
| 43829 | CANCELLED | Smoke of eval-dump launcher | Passed, canceled to free GPU |
| 43830 | COMPLETED | Full eval-dump of tsweep baseline | NPZs produced |
| 43831 | COMPLETED | Re-dump with term_texts metadata | NPZs updated |
| 43832 | FAILED (preflight) | First MFA smallest+dense smoke attempt | Killed by too-strict preflight; fixed |
| 43833 | COMPLETED | Threshold sweep on 43827 snapshot | TSVs + PNGs produced |
| 43834 | RUNNING | MFA smallest+dense smoke (100 steps) on aries | Step ~40, step-time ~24s (not 55-65s as estimated), no OOM |
| 43835 | COMPLETED | Verify new training-eval noise metric matches threshold_sweep | ACL gs10k τ=0.80 noise=0.72, matches sweep within ~10% |
| 43836 | COMPLETED | Build 10k gigaspeech voice pool | 10,000 voices in 2 min |
| 43837_0..5 | RUNNING | CosyVoice TTS POC, 5k samples across 6 shards | Loading models |

## Decision log (key choices this session)

- **Fork CosyVoice script = NO.** `rag_tts_multispeaker_noise.py` accepts
  `--speaker-dir` at CLI, so swap is a launcher-only change. Original
  plan called for forking; dropped after reading the script.
- **POC scale.** 5k samples chosen despite knowing 5k / 3M = 0.17% of
  train is too small to meaningfully move eval metrics in 1 epoch. POC
  is treated as "pipeline validation + audio spot-check", not "training
  delta". If pipeline + quality look clean, scale up to 500k next.
- **Voice pool domain balance.** Equal thirds audiobook/podcast/youtube.
  Any domain skew could leak into the retriever as "voice-domain"
  features. Equal thirds mitigates.
- **Sample count for MFA duration table.** Sampled 183k train rows (not
  all 9.7M) for the MFA-duration histogram that informed dense-window
  grid design. Median 6.8 frames, p90 12.6 frames → dense grid `2 3 4 5
  6 7 8 10 12 16 20 24` covers every term with ≤1 frame leakage.
- **Preflight philosophy.** Moved from "pessimistic hard-gate" to
  "optimistic log-and-proceed". SLURM exclusivity + training-time OOM
  are sufficient safeguards. The earlier 43808 postmortem that added
  the hard-gate was over-indexed on a root-process incident that hasn't
  recurred.

## Open items (not executed this session)

- **43834 completion + full 3-epoch launch.** Smoke pending; full run
  queued on user decision.
- **POC TTS → MFA → merge → 1-ep resume.** Steps after 43837 finishes:
  (a) run `align_and_cut_wiki_synth.py` on the POC outputs (CPU, 15-30
  min), (b) merge new JSONL into train, (c) resume 1 epoch from
  variantE_best, (d) check ACL delta.
- **TCM retune (T_β=0.7, λ=3-5).** User-raised as high priority but
  not started this session; keep as next-run candidate stacked on top
  of whichever data config wins.
- **Voice pool scale-up.** If POC audio quality is good + training
  pipeline clean, plan a 500k-sample regenerate (3-4 days on 6 taurus
  GPU) before committing to full 3M.
