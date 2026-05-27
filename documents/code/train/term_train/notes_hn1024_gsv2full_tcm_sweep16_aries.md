# Full GSV2 k1024 TCM pair/weight continuation sweep

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa` / `train`
- **Variant family**: `hn1024_gsv2full_tcm_pair_w`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcm_sweep16_8gpu_aries.sh`
- **Warm-start baseline**: `hn1024_gsv2full_tcmoff_ep3`

## Hypothesis

Continuing one epoch from a strong TCM-off checkpoint should reveal whether TCM
mainly needs a stricter operating pair or a larger auxiliary weight. Binding
positive and negative thresholds into four ordered pairs should capture the main
calibration tradeoff with 16 runs instead of a full 80-run Cartesian grid.

## Background / Motivation

Previous TCM scouts suggest ACL6060 can overstate confidence when the threshold
choice is tuned on the same domain. This sweep uses only dev metrics and an
untrained P31 gs10000 glossary for checkpoint selection, with ACL disabled in
the training jobs.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr
- Diff:
  - resume checkpoint: full GSV2 k1024 TCM-off epoch-2 checkpoint with AdamW state
  - continuing horizon: one additional epoch (`start_epoch=3`, `epochs=4`)
  - threshold pairs: `(0.85,0.70)`, `(0.80,0.60)`, `(0.75,0.50)`, `(0.70,0.40)`
  - shared branch weights: `tcm_pos_loss_weight=tcm_neg_loss_weight in {1,2,4,8}`
  - TCM loss form/reduction/scope: `hinge` / `mean_viol` / `all`
  - eval selection: best checkpoints track `eval_dev/*`; ACL6060 is disabled

## Expected metrics

Rank settings by dev `recall@10_gs10000`, `topk10_filtered_recall@tau_0p80_gs10000`,
neighboring tau-filtered recall, and no-term noise. A winning setting should
improve calibration at `tau=0.75/0.80` without a large drop in dense gs10000
recall or a large increase in no-term avg-kept noise.

## Verdict

PENDING: update after the 16 continuing runs finish and are compared at
best-step on dev metrics.
