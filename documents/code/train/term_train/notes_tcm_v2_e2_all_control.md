# E2 — TCM-v2 full-grid control from `tys70s0y`

Matched control for E1 from the "Best TCM Exploration" plan: keep the same
`hinge + mean_viol` formulation, the same split positive/negative weights, and
the same thresholds, but leave the negative TCM branch on the legacy full
negative grid. This isolates whether any gain comes from the new
candidate-aware scope rather than simply from switching away from
`squared_hinge` or splitting the branch weights.

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k512_tcmv2_allscope_smallest_dense_normAGGR_6gpu`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_tcm_v2_allscope_6gpu_aries.sh`
- **Baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

If E1 wins mainly because the negative branch now sees the same hard top-list
competitors that matter at inference time, then the all-negative control should
underperform E1 on `noterm_noise@top10_tau*_gs10000` and/or pay a larger recall
cost at the same locked `tau*`.

## Background / Motivation

The current TCM implementation penalizes the full negative matrix, but
deployment only ever exposes the LLM to the filtered top-`k` candidate list.
E2 is required so that a win in E1 can be attributed to fixing this scope
mismatch rather than to the more cosmetic changes (`hinge`, split weights).

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **Diff**:
  - code: same TCM-v2 hooks as E1
  - hparam `tcm_loss_form`: `squared_hinge` -> `hinge`
  - hparam `tcm_reduction`: unchanged at `mean_viol`
  - hparam `tcm_pos_loss_weight`: shared `1.0` -> `0.25`
  - hparam `tcm_neg_loss_weight`: shared `1.0` -> `1.0`
  - hparam `tcm_neg_scope`: remains `all` (legacy full-grid behavior)
  - hparam `tcm_neg_topk`: `0` (inactive by design)
  - hparam `tcm_pos_threshold`: `0.85` -> `0.74`
  - phase-0 tau lock: `tau*=0.80` on ACL6060 `gs10000` dense sweep
  - hparam `tcm_neg_threshold`: `tau* - 0.02 = 0.78`
  - hparam `tcm_warmup_steps`: `0` -> `100`
  - data / HN policy / batch size / GPUs / LR / temperature / MFA recipe:
    unchanged from `tys70s0y`

## Expected metrics

E2 is a control, not the preferred win condition:

- if E2 matches E1, then `tcm_neg_scope=topk` is probably not the main factor
- if E2 is clearly worse on `noterm_noise@top10_tau*_gs10000` or recall at the
  locked `tau*`, then the scope change is likely the real contributor
- either way, `eval_dev/*` and `eval_acl6060/*` at the locked `tau*=0.80` should
  remain better than the failed full-grid `lambda=5` direction

## Verdict

Archived on 2026-04-24 before launch. The queued SLURM job was cancelled when
the experiment order changed from TCM-first to HN-depth-first, so this control
never produced a training run.
