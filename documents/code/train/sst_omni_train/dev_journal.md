# train/sst_omni_train / dev_journal.md

Active notebook for Speech-LLM (Qwen3-Omni) adversarial / density training
runs. Distill stabilized recipes into `train/sst_omni_train/README.md` when
a training recipe is locked.

---

## 2026-03-28 — Rank ablation r=32: OOM on 2x A6000 (INFEASIBLE with all-linear LoRA)

Context: part of the 8h audit + rank-ablation exploration. Full report at
`documents/data/phase456_orchestration/REPORT_audit_and_rank_ablation.md`.

### Finding

`d5 r=32 alpha=32 target_modules=all-linear` cannot fit on taurus 2x A6000 (48GB
each) using `run_speech_llm_4gpu_maxsim.sh` (NPROC=2, EP=2). Tested three
`max_length` values; all three jobs OOM at iteration 2 during the fused
vocab-parallel cross-entropy logits allocation.

| attempt | slurm job | max_length | iter-1 peak | crash site |
|---|---|---|---|---|
| 1 | 43753 | 4096 (default) | 45.8 GiB | `audio_tower.conv2d1` gelu |
| 2 | 43755 | 3072 | 45.0 GiB | fused-CE, needs 1.73 GiB, 636 MiB free |
| 3 | 43756 | 2048 | 44.4 GiB (45.4 GiB reserved) | fused-CE, needs 1.16 GiB, 68 MiB free |

r=16 fits on the same recipe because LoRA A/B matrices (and grads) are half
the size at every MoE-expert linear (there are hundreds of MoE linears).

### Follow-ups

1. Extend `run_speech_llm_4gpu_maxsim.sh` with a 4-GPU path
   (`NPROC_PER_NODE=4`, `EP=4`) behind `ENABLE_4GPU=1`. This shards MoE
   experts across 4 ranks and should cut the activation footprint nearly
   in half.
2. Alternatively, add a `TP_SIZE=2` knob to shard the output head logits
   (that is what OOM'd in attempts 2-3).
3. `r=64` was cancelled (slurm 43754); resubmit after 1-2 land.

### Artifacts

- `documents/code/tools/run_8h_audit_rank_neg.sh` — orchestrator (now with
  `T1_MAX_LENGTH` / `T2_MAX_LENGTH` env knobs).
- `documents/code/train/sst_omni_train/run_rank_ablation_sbatch.sh` — single-
  variant sbatch wrapper (reads `LORA_RANK_OVERRIDE`, `MAX_LENGTH_OVERRIDE`,
  `SAVE_BASE_OVERRIDE`, `DATASET_PATH_OVERRIDE`).
- Run root: `/mnt/gemini/data1/jiaxuanluo/logs/run_8h_audit_rank_neg_20260419_065725`

---

## 2026-03-28 — d5_cap_adv_half (half-ratio adversarial)

Rationale: see `simuleval/dev_journal.md` — d5_cap_adv had BLEU -2.37 and
AS-WER +4.09 vs no-cap d5 baseline, with only marginal per-paper TERM_ACC
gain. Try halving the adversarial perturbation ratio to see if the BLEU
cost falls off faster than the TERM_ACC recovery.

### Config

Reuses `run_adversarial_train_sbatch.sh` (now has 3 variants). New variant
entry:

```
half:5_cap_adv_half:\
  /mnt/gemini/data1/jiaxuanluo/adversarial/train_maxsim_varlen_d5_cap_adv_half.jsonl:\
  /mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv_half:\
  2:aries
```

Identical to Phase 4 `experiment`:
- LoRA rank 16
- EP=2 / 2 GPUs per job
- aries partition (taurus GPU 5/6/7 are free on taurus too, but aries is
  less contended right now)
- `WANDB_MODE=offline`
- dataset size 39.4 MB (= same as d5_cap / d5_cap_adv; rebuild_termmap
  preserves schema)

### Submission

```
bash documents/code/train/sst_omni_train/run_adversarial_train_sbatch.sh half
```

Result: `sbatch → 43725`, state `R` on `aries` within seconds.
Logs: `/mnt/gemini/data1/jiaxuanluo/logs/43725_train_5_cap_adv_half.{out,err}`

### Expected deliverables

- `/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv_half/r16/v*-hf/`
  (HF export) — feed into `run_phase5_model_eval.sh` with `DENSITY_TAG=5`
  and `MODEL_NAME=d5_cap_adv_half`.

### 2026-03-28 23:44 — aborted

`43725` was `scancel`-ed at iteration **150 / 585** (3h30m elapsed).

Reason:
- aries node this time ran at 67s/iter vs the 43715 reference at 15s/iter
  — **4.4x slowdown**, projected ETA 10–11h instead of 3h. Cause: shared-
  resource contention on the aries node (co-tenant `43724` and other jobs
  stressing PCIe / memory-bandwidth; `NCCL_P2P_DISABLE=1` routes GPU–GPU
  through CPU, which magnifies the hit).
- Loss trajectory at iter 150 (0.70) matched the 43715 trajectory at iter
  150 (0.71), so the training itself was fine; only the wall-clock was bad.
- The 3-way (d5 / d5_cap / d5_cap_adv) analysis in
  `simuleval/dev_journal.md` already shows the adversarial path is a net
  loss vs the no-cap baseline (BLEU -2.37, AS-WER +4.09, full-glossary
  TERM_ACC unchanged). Continuing the half-ratio run was not expected to
  flip that conclusion, only to trade some BLEU back for some per-paper
  TERM_ACC.

Cleanup performed:
- `scancel 43725` → state `CANCELLED by 1039`, elapsed `03:30:01`
- `kill 3256854` (orchestrator)
- removed unfinished save dir
  `/mnt/taurus/data2/jiaxuanluo/speech_llm_density_ablation/d5_cap_adv_half/`
- removed unused training JSONLs on gemini:
  `train_cleaned_with_retriever_results_varlen_adv_half.jsonl`,
  `train_maxsim_varlen_d5_cap_adv_half.jsonl`,
  `perturbation_stats_half.json`
- commented out (not deleted) the `half:` row in
  `run_adversarial_train_sbatch.sh` so the recipe is preserved for a
  future retry on a non-contended node.

No new HF export was produced, so no Tier-3 (`general/`) changes needed.
Tier-2 distillation lives in `simuleval/README.md`.
