# Record: TCM (Threshold-Consistent Margin) auxiliary loss for retriever training

Date: 2026-04-18
Author: agent

## Goal

Replace the post-hoc confidence-head pipeline with an absolute-threshold loss
baked directly into retriever training.  Use the sim-distribution diagnostic
on the Config C retriever to motivate the thresholds:

- Positive cos-sim distribution shows a long low-sim tail on the ACL6060 domain
  (mean 0.74, p10 0.54) with heavy overlap against negatives.
- `S_neg_top1_pure` (top-1 neg sim for no-term chunks against a 10k-term bank)
  reaches p99 ~ 0.95 on both gigaspeech and ACL6060 -> InfoNCE alone does not
  constrain absolute negative scores.

Plan: add TCM loss (Zhang et al., ICLR 2024) as an auxiliary term added to
InfoNCE during retriever training, penalizing positives below `T_beta` and
negatives above `T_alpha`.

## Files added / modified

### Modified

- `documents/code/train/term_train/qwen3_glossary_neg_train.py`
  - Added TCM Configuration block (defaults: `lambda=0.0`, `T_beta=0.7`,
    `T_alpha=0.4`, `squared_hinge`, reduction `mean_viol`).
  - `compute_masked_contrastive_loss` now accepts TCM kwargs and returns a
    dict `{total, infonce, tcm_pos, tcm_neg, tcm_pos_viol_rate,
    tcm_neg_viol_rate, pos_sim_mean, neg_sim_mean}`.
  - `gradcache_train_step` returns `(loss_outputs, hard_neg_count)` instead
    of `(total_loss, hard_neg_count)`.  Both the grad-cache and non-grad-cache
    callers updated to use `loss_outputs["total"]` for `.backward()`.
  - Added fallback in grad-cache phase-3 `.no_sync()` call so single-GPU
    smoke testing works (pre-existing bug where plain nn.Module has no
    `no_sync`; only DDP-wrapped models do).
  - Train-loop logger + WandB report TCM components (`loss_infonce`,
    `loss_tcm_pos`, `loss_tcm_neg`, `tcm_pos_viol_rate`,
    `tcm_neg_viol_rate`, `pos_sim_mean`, `neg_sim_mean`).
  - New CLI flags: `--tcm_loss_weight`, `--tcm_pos_threshold`,
    `--tcm_neg_threshold`, `--tcm_loss_form`, `--tcm_reduction`.

### Added

- `documents/code/train/term_train/run_tcm_smoke_taurus.sh`
  - Single-GPU smoke test (train_limit=2000, batch=256, epochs=6) that
    verifies the TCM code compiles + runs, prints TCM components, and does
    forward/backward through GradCache correctly.
  - Smoke output: `/mnt/gemini/data2/jiaxuanluo/tcm_smoke_logs/smoke_v4.log`
    ```
    step=1420  loss=0.541  infonce=0.529  tcm_pos=0.111  tcm_neg=0.011
    step=1440  loss=0.188  infonce=0.179  tcm_pos=0.074  tcm_neg=0.011
    pos_sim_mean drops 0.857 -> 0.839
    neg_sim_mean drops 0.045 -> 0.040
    ```
    TCM contributes ~2-4% of total loss (with `lambda=0.1`, `mean_viol`),
    positive violations shrink (0.111 -> 0.074), negative violations stay
    steady (0.011 - expected since mean is over the rare hard negatives).

- `documents/code/train/term_train/run_3variant_1m_aries_gc12k_maxsim_mfa_tcm.sh`
  - Aries 8-GPU SLURM launcher.  Mirrors `run_3variant_1m_aries_gc12k_maxsim_mfa.sh`
    exactly, with:
      - TCM flags (`lambda=0.1`, `T_beta=0.7`, `T_alpha=0.4`, squared-hinge,
        `mean_viol` reduction).
      - `RESUME_PATH=""` (from-scratch training; old step_300 ckpt no longer exists).
      - `VERSION` suffix changed to `_maxsim_mfa_tcm_v1_fromscratch`.
      - Fully-qualified cross-node paths (`/mnt/aries/data4/...`) per user rule.
  - Not yet submitted - pending user review.

## Data locations

- Diagnostic used to choose thresholds:
  `/mnt/gemini/data2/jiaxuanluo/retriever_sim_diag/full_v1/`
- Smoke logs:
  `/mnt/gemini/data2/jiaxuanluo/tcm_smoke_logs/smoke_v4.log`
- Expected TCM training output:
  `/mnt/aries/data4/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.03_3var_clean_gc_wr1000k_m0.1_maxsim_mfa_tcm_v1_fromscratch.pt`

## TCM loss formula

For cosine similarities `s_ij = cos(chunk_i, term_j)` in the masked batch:

```
L_InfoNCE  = masked multi-positive InfoNCE (existing).
L_TCM_pos  = sum_{(i,j) in P}      relu(T_beta  - s_ij)^2  /  num_violating_pos_pairs
L_TCM_neg  = sum_{(i,j) in N_valid} relu(s_ij - T_alpha)^2 /  num_violating_neg_pairs
L_total    = L_InfoNCE + lambda_TCM * (L_TCM_pos + L_TCM_neg)
```

- P = positive pairs (same group id, both valid).
- N_valid = all non-positive, non-FN pairs with valid term (in-batch +
  glossary bank).
- `mean_viol` reduction: denominator is violation count to keep gradient
  magnitude non-trivial at batch=12k (compare `mean_all` which dilutes to
  essentially zero).

## Open items / next steps

1. User to review `run_3variant_1m_aries_gc12k_maxsim_mfa_tcm.sh` Configuration
   block, then `sbatch` submit.
2. During training: monitor WandB for `train/loss_tcm_pos`, `train/loss_tcm_neg`,
   `train/tcm_pos_viol_rate`, `train/tcm_neg_viol_rate`, `train/pos_sim_mean`,
   `train/neg_sim_mean` trajectories.  Expect tcm_pos to drop, tcm_neg to drop
   (steady at first, then slowly), pos_sim_mean to rise, neg_sim_mean to fall.
3. Post-train: re-run `documents/code/offline_evaluation/retriever_sim_distribution.py`
   on the best TCM checkpoint to compare against Config C and verify the
   distribution has tightened around the target thresholds.
4. Ablation sweep (future): `lambda in {0.05, 0.1, 0.2}`, `mean_all` vs
   `mean_viol`, separate positive-only vs negative-only TCM.
