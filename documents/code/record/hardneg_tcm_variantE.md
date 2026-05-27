# Variant E: bank-mined hard-negatives + TCM (no HCL)

Last updated: 2026-04-20

## Goal

Fix the data-side root cause of TCM's weak neg-side effect in the earlier
2x2 ablation (A/B/C/D): in-batch random negatives from a 12k-term training
pool never contained genuine in-domain hard negatives, so `neg_sim_mean` at
convergence sat at ~0.04 and `tcm_neg` barely activated.  Variant E plugs
the data-side gap with explicit top-K bank mining and drops HCL because the
in-domain hard negatives already reshape the InfoNCE softmax.

## Code changes

### `documents/code/train/term_train/qwen3_glossary_neg_train.py`

`NegativeTermBank.refresh()` was rewritten to shard term encoding across DDP
ranks (ceil-shard pattern + zero-padded `torch.distributed.all_gather` +
trim).  Under DDP every rank now encodes `ceil(N / world_size)` terms and
assembles an identical CPU copy of the full bank.  Single-GPU path is
preserved verbatim.  Fails loud (`ValueError`) if `world_size > bank_size`.

Measured refresh overhead on aries 8x A6000 at bank size 43k: ~1-2 s
amortized (vs ~6.5 s single-GPU at 22k, so roughly 8x speedup as expected
once the bank clears a few batch-sized shards per rank).

### `documents/code/train/term_train/smoke_hardneg_tcm_aries.sh`

Existing 1-GPU smoke; reused to produce the HCL=0 reference curve (job
43763).

### `documents/code/train/term_train/smoke_hardneg_tcm_hcl1_aries.sh` (new)

Identical to the HCL=0 smoke except `HCL_BETA=1.0`.  Produced the HCL=1
comparison curve (job 43764) under strictly matched LR schedule, bank
refresh cadence, and dataloader seed (DistributedSampler default seed=0).

### `documents/code/train/term_train/smoke_hardneg_tcm_8gpu_aries.sh` (new)

8-GPU short-wall-time smoke to exercise the new DDP-parallel refresh path
before burning the full 2h slot.  Job 43765 completed cleanly with
refreshes at step 1 and step 5.

### `documents/code/train/term_train/run_hardneg_tcm_aries.sh` (new)

Full 8-GPU, 5-epoch, from-scratch launcher for variant E.  Wiring:
- `MARGIN=0.0`, `HCL_BETA=0.0`
- `TCM_LOSS_WEIGHT=1.0`, `T_beta=0.85`, `T_alpha=0.25`,
  `TCM_REDUCTION=mean_viol`, `TCM_LOSS_FORM=squared_hinge`
- `HARD_NEG_K=64`, `NEG_BANK_REFRESH_STEPS=50`,
  `HARD_NEG_GLOSSARY=.../wiki_hard_neg_disjoint.json`
- `PER_GPU_BATCH=1536`, `BATCH_SIZE=12288`, `GRAD_CACHE_CHUNK_SIZE=256`
- `MAX_TRAIN_SECONDS=0`, `EPOCHS=5` (natural 5-epoch cosine schedule;
  `max_train_seconds=0` bypasses the EPOCHS<=2 alignment guard),
  `SAVE_DIR=/mnt/gemini/home/jiaxuanluo/train_outputs`
- slurm `--time=14:00:00` (aries has `infinite` partition cap; estimated
  runtime ~9.5 h at ~13 s/step * 530 steps * 5 epochs).

At full scale the bank is **1.296M unique terms** (1.287M training-set GT
terms + 8.3k disjoint wiki hard negs), so the parallel refresh's ~8x
speedup actually matters (~30-40 s / refresh on 8 GPUs vs ~5 min serial).
Refresh cadence of 50 steps keeps amortized overhead under 5% wall-time.

### `documents/code/train/term_train/qwen3_glossary_neg_train.py` -- TCM eval metrics

Added `_compute_tcm_threshold_metrics()` and wired it into
`run_sample_eval()` (both the base-bank block and the per-gs expanded-bank
block).  For every eval run we now report, at both `T_beta` (0.85, strict)
and `T_alpha` (0.25, loose):

- `tcm_precision@{tbeta,talpha}` = TP / total_accepted_cells across the
  full query x bank matrix
- `tcm_recall@{tbeta,talpha}` = TP / N (fraction of queries whose gt
  score clears the threshold)
- `tcm_f1@{tbeta,talpha}`
- `tcm_pass_rate@{tbeta,talpha}` = fraction of queries with at least one
  candidate crossing the threshold (natural downstream keep-rate)
- `tcm_{tbeta,talpha}_value` (echoed thresholds for sanity)

These land under the normal `eval_dev/` and `eval_acl6060/` prefixes and
per-gs suffixes (e.g. `eval_acl6060/tcm_precision@tbeta_gs1000`) so TCM's
calibration effect is directly visible in the wandb dashboard alongside
the existing `recall@10` curves.

## HCL drop decision (smoke A/B at identical config)

Jobs 43763 (HCL=0) vs 43764 (HCL=1), same LR schedule, same refresh=10,
same batch order, single GPU bs=512, 54 total optimizer steps.

| Step | metric       | HCL=0  | HCL=1  | delta |
| ---- | ------------ | ------ | ------ | ----- |
| 20   | infonce      | 5.6625 | 6.1090 | +0.45 (loss scale only) |
| 20   | pos_sim      | 0.501  | 0.509  | +0.008 |
| 20   | neg_sim_mean | -0.030 | +0.025 | +0.055 |
| 20   | score_gap    | 0.531  | 0.484  | -0.047 (HCL=0 wider) |
| 40   | infonce      | 5.5223 | 5.9364 | +0.41 (loss scale only) |
| 40   | pos_sim      | 0.580  | 0.573  | -0.007 |
| 40   | neg_sim_mean | 0.168  | 0.175  | +0.007 |
| 40   | score_gap    | 0.412  | 0.398  | -0.014 (HCL=0 wider) |

HCL=1 only inflates the absolute loss via the `exp(beta * s)` reweighting
of the negative-sum term; the learned representations (`pos_sim`,
`neg_sim_mean`) are essentially identical.  Under bank mining, dropping
HCL is the simpler, lower-variance choice.

### `documents/code/train/term_train/qwen3_glossary_neg_train.py` -- chunked hard-neg mining

Job 43768 OOM'd at the first `mine_hard_negatives` call on a 1.296M bank:
`torch.matmul(speech_embs, text_embs.T)` materialised `[B=1536, W~180, N=1.3M]`
fp32 ~178 GiB on a 48 GiB A6000.  The earlier 8-GPU smoke (43765) did not
catch this because `train_limit=98304` capped the bank at ~33k.

Fix: `NegativeTermBank.mine_hard_negatives` now iterates over bank chunks
of `DEFAULT_HARD_NEG_MINE_CHUNK = 32768` terms and keeps a rolling top-K
(values + global indices) across chunks.  Mining also casts speech +
bank to bf16 for the sim matmul; the returned embeddings are still
fetched from the fp32 master bank so loss precision is unaffected.
Peak activation drops from ~1.2 TB to ~18 GB per chunk on 8x A6000,
per-mine wall-time ends up well under 1 s amortized.

## Submitted

- `sbatch run_hardneg_tcm_aries.sh` -> job **43769** (running, 5-epoch).
  Steady state ~15.5 s/step; 8h elapsed @ step 1300/2650 = 49%.  SLURM
  cap of 14h cannot be extended by the user
  (`scontrol update timelimit`: Access denied) so the run will complete
  roughly 3.5-4 epochs before being killed.  `_best.pt` and
  `_best_acl6060_gs10000.pt` keep being refreshed (last updates at step
  1080 / 1240 at 08:29 / 09:15), so the best-metric checkpoints remain
  current even if the last epoch is cut short.
- Mid-training health (step 1280, epoch 2):
  - train `infonce` 3.76 -> 1.68, `pos_sim` 0.805 -> 0.893
    (already over `T_beta=0.85`), `neg_sim` 0 -> -0.05.
  - `eval_dev`: `recall@10_gs10000` 0.9477, `tcm_P/R@tbeta` 0.71 / 0.82,
    `tcm_pass@tbeta` 0.89 (TCM calibration clearly working in-domain).
  - `eval_acl6060`: `recall@10_gs1000` 0.9333, `recall@10_gs10000`
    0.8698 (OOD), `score_gap` 0.10 vs 0.045 at step 320.
- Resume launcher `run_hardneg_tcm_resume_aries.sh` prepared (picks the
  latest `..._epoch_N.pt` by mtime, keeps optimizer + scheduler state so
  the remaining cosine tail continues instead of re-warming).  Submit
  after 43769 dies if the eval trend is still visibly rising.
- Supersedes 43766 (2h / EPOCHS=1, cancelled because a wall-time cap
  forced premature LR decay), 43767 (cancelled to pick up the new
  TCM-gated eval metrics), and 43768 (OOM in `mine_hard_negatives` at
  1.296M-bank scale, fixed above).
- W&B run name:
  `variantE_q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_variantE_hardneg_tcm_ep5`.

Compared at the 5-epoch endpoint against the rerun ablation A (InfoNCE
baseline) and D (InfoNCE + HCL + TCM at T_beta=0.7 / T_alpha=0.4) on
`acl6060/recall@10` and, new this run, `acl6060/tcm_precision@tbeta` /
`acl6060/tcm_recall@tbeta`.
