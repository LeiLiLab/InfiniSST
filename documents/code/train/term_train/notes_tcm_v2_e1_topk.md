# E1 — TCM-v2 candidate-aware neg scope (`topk=32`) from `tys70s0y`

Primary ablation from the "Best TCM Exploration" plan: keep the `tys70s0y`
retriever recipe fixed, switch TCM to `hinge + mean_viol`, split positive and
negative weights, and restrict the negative TCM branch to the hardest per-row
`topk=32` negatives so the auxiliary loss matches the real deployment path
(`top10` retrieval, then tau filtering).

- **Family / data / task**: `sst_ood_hardneg` / `3variant_1m_mfa` / `train`
- **Variant tag**: `hnps_k512_tcmv2_topk32_smallest_dense_normAGGR_6gpu`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_tcm_v2_topk32_6gpu_aries.sh`
- **Baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

If ACL boundary failures are mostly real high-scoring noise near the top of the
candidate list, then applying the negative TCM branch only to the hardest
per-row negatives should cut `noterm_noise@top10_tau*_gs10000` more efficiently
than the legacy full-grid TCM, while keeping filtered recall flat. A light
positive branch (`0.25`) should preserve recall better than the old symmetric
shared-`lambda` setup.

## Background / Motivation

The ACL audit for `tys70s0y` showed `323 / 397` boundary rows are `clear_noise`,
and the critical `gt_missing_or_outranked` subset is `89 clear_noise / 16 very
similar / 0 likely_false_negative`. That means the highest-ROI TCM change is
not another blind `alpha/beta/shared_lambda` sweep on the full negative grid;
it is to align the neg-side TCM target with the same near-top competitors that
actually survive into deployment.

## What changed vs baseline

- **Baseline run URL**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y
- **Diff**:
  - code: add backward-compatible TCM-v2 hooks in `qwen3_glossary_neg_train.py`
    for `--tcm_pos_loss_weight`, `--tcm_neg_loss_weight`,
    `--tcm_neg_scope`, and `--tcm_neg_topk`
  - hparam `tcm_loss_form`: `squared_hinge` -> `hinge`
  - hparam `tcm_reduction`: unchanged at `mean_viol`
  - hparam `tcm_pos_loss_weight`: shared `1.0` -> `0.25`
  - hparam `tcm_neg_loss_weight`: shared `1.0` -> `1.0`
  - hparam `tcm_neg_scope`: legacy full negative grid -> `topk`
  - hparam `tcm_neg_topk`: new -> `32`
  - hparam `tcm_pos_threshold`: `0.85` -> `0.74`
  - phase-0 tau lock: `tau*=0.80` on ACL6060 `gs10000` dense sweep
  - hparam `tcm_neg_threshold`: `tau* - 0.02 = 0.78`
  - hparam `tcm_warmup_steps`: `0` -> `100`
  - data / HN policy / batch size / GPUs / LR / temperature / MFA recipe:
    unchanged from `tys70s0y`

## Expected metrics

The acceptance bar is the locked `tau*=0.80` chosen in phase 0:

- `eval_acl6060/topk10_filtered_recall@tau_0p80_gs10000`: at least flat vs
  `tys70s0y`
- `eval_acl6060/noterm_noise@top10_tau_0p80_gs10000`: lower than `tys70s0y`
- `eval_dev/topk10_filtered_recall@tau_0p80_gs10000`: no material regression
- boundary audit after training: fewer `clear_noise` rows, especially in the
  `gt_missing_or_outranked` subset

## Verdict

Archived on 2026-04-24 after a strategy pivot: TCM is deferred until the
non-TCM HN-depth sweep fixes the base retriever recipe. This run was cancelled
at step `140` (WandB `ailr03qx`) and flipped to `status:failed` only for
bookkeeping; no scientific conclusion should be drawn from it.
