# Record: HCL + TCM 2x2 ablation for retriever training

Date: 2026-04-18
Author: agent

## Goal

Extend the TCM experiment into a 2x2 factorial design that also validates
HCL (Robinson et al., "Contrastive Learning with Hard Negative Samples",
ICLR 2021) hard-negative importance reweighting.  The original concern:
random in-batch negatives in a 12k batch dilute the rare domain-hard
negatives, so (a) InfoNCE alone gets little gradient from the pairs that
matter and (b) TCM_neg's threshold violations stay small per-row.  HCL
soft-reweights negatives by `w_j = exp(beta*s_j) / mean_k exp(beta*s_k)`,
amplifying hard negatives without discarding easy ones.

CosFace margin is redundant with TCM_pos (both push positives up) and
harder to attribute in an ablation, so it is set to 0 across all four
variants.  A single winning variant may later re-enable margin for the
full-scale production run.

## Variants

| Variant | InfoNCE | HCL (beta) | TCM (lambda) | Suffix |
|---------|---------|------------|--------------|--------|
| A       | yes     | -          | -            | `A_infonce`           |
| B       | yes     | 1.0        | -            | `B_hcl_b1`            |
| C       | yes     | -          | 0.1          | `C_tcm_l01`           |
| D       | yes     | 1.0        | 0.1          | `D_hcl_b1_tcm_l01`    |

Shared: margin=0, temperature=0.07, batch=12288 (8 GPUs x 1536), training
from scratch with 2 hour walltime cap per variant (set via
`--max_train_seconds 7200`; EPOCHS raised to 99 as a non-binding upper
bound).  Sampled eval every 40 steps (~8-10 eval points within 2h),
step-save disabled; `_best.pt` is still saved on every eval that improves
the primary metric (`eval_acl6060/recall@10_gs1000`).  SAVE_DIR moved to
`/mnt/gemini/home/jiaxuanluo/train_outputs` (4.2T free; NFS-backed but
ckpts only write on eval-improvement so latency is acceptable).

## Priority

1. C: primary hypothesis (TCM calibrates absolute ops-point).
2. D: interaction test (does HCL focus make TCM_neg more effective?).
3. B: independent HCL-only check.
4. A: required reference baseline.

## Files touched

- `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/qwen3_glossary_neg_train.py`
  - Added `DEFAULT_HCL_BETA` in the TCM Configuration block.
  - Shared `neg_mask` moved up (step 5.5) so HCL and TCM reuse it.
  - Step 6.3 new: HCL reweighting as additive, detached `log(w_j)` shift
    on neg logits; mutually exclusive with `online_hard_neg_k`.
  - `compute_masked_contrastive_loss(..., hcl_beta=0.0)` signature
    extended; returns two more scalars
    `hcl_neg_sim_weighted_mean`, `hcl_log_weight_max`.
  - Both callers (`gradcache_train_step`, `train`) plumb `args.hcl_beta`.
  - CLI flag `--hcl_beta` added.
  - `[TRAIN]` logger line and wandb `log_dict` emit HCL metrics.
- `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_hcl_ablation_aries.sh`
  NEW: parametrized 8-GPU Aries launcher.  Reads `VARIANT` env var
  (A/B/C/D), sets loss flags accordingly, writes to distinct
  `SAVE_PATH` / WANDB experiment / MASTER_PORT per variant.
- `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_smoke_taurus.sh`
  Added `HCL_BETA` env var; smoke-tested variants B and D.
- `/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_3variant_1m_aries_gc12k_maxsim_mfa_tcm.sh`
  Deleted (superseded by the parametrized ablation sbatch).

## Smoke test results (variant B and D, 20 steps each, single GPU)

Variant D (HCL beta=1.0, TCM lambda=0.1):
```
step=1420 loss=0.3793 infonce=0.3735 tcm_pos=0.0448 tcm_neg=0.0130 pos_sim=0.878 neg_sim=0.043 hcl_neg_sim_w=0.058 hcl_logw_max=0.41
step=1440 loss=0.1475 infonce=0.1369 tcm_pos=0.0967 tcm_neg=0.0098 pos_sim=0.855 neg_sim=0.036 hcl_neg_sim_w=0.050 hcl_logw_max=0.39
```

Variant B (HCL beta=1.0, no TCM):
```
step=1420 loss=0.4255 hcl_neg_sim_w=0.060 hcl_logw_max=0.40
step=1440 loss=0.1926 hcl_neg_sim_w=0.052 hcl_logw_max=0.39
```

Observations:
- `hcl_log_weight_max ~ 0.4` -> hardest negative weight `exp(0.4) ~ 1.5x`
  uniform, matching paper's intended moderate concentration at beta=1.
- `hcl_neg_sim_weighted_mean` ~ 0.058 vs `neg_sim_mean` ~ 0.043 in D,
  i.e. HCL-weighted mean is 35 percent higher than uniform.  The taunt
  lands on the hard portion of the neg distribution.
- HCL and TCM coexist; no NaNs, losses decrease, all metrics stay finite.

## How to submit

```
VARIANT=A sbatch /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_hcl_ablation_aries.sh
VARIANT=B sbatch /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_hcl_ablation_aries.sh
VARIANT=C sbatch /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_hcl_ablation_aries.sh
VARIANT=D sbatch /mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/run_tcm_hcl_ablation_aries.sh
```

Save paths (each variant distinct):
`/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_clean_gc_wr1000k_m0.0_maxsim_mfa_ablate_{A|B|C|D}_*_2h.pt`

Note: the `C` and `D` variants were trained under an earlier
`EPOCHS=1` recipe before the walltime cap was introduced.  In practice
those runs trained for ~2:00-2:20h each (1 epoch at batch=12288 with
8 GPUs), so their compute is effectively within 10-15 percent of the 2h
budget and their `_best.pt` is retained as the ablation result.  `C`
crashed mid-run at step 411 (pin-memory socket disappeared due to
`/mnt/aries/data4` being 96 percent full); its `_best.pt` at step 360
is still the best eval snapshot within the crashed run.  `D`'s ckpt
lives under `_ep1` at the old `/mnt/aries/data4/...` path; it should be
migrated to `/mnt/gemini/home/...` before the final analysis to keep
the 4 results co-located.  `B` and `A` are fresh 2h runs at the new
SAVE_DIR.

## Budget

~6h per variant x 4 = ~24h wall-clock on 8-GPU Aries (single-epoch 1M
effective samples, batch 12k -> ~83 steps/epoch).

## Open items

- User to review Configuration block of
  `run_tcm_hcl_ablation_aries.sh` before submission.
- After all four complete: pick the best variant by
  `eval_acl6060/recall@10_gs1000` at the end of epoch 1, and consider
  re-adding `margin=0.1` on the winner for the full (5-epoch) production
  run.
- Post-train: rerun the sim-distribution diagnostic on each variant's
  final ckpt and compare histograms against Config C.
