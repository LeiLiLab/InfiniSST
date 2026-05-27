# TCM Anchor Score Distribution Eval

## Hypothesis

The converged TCM-off step-2650 checkpoint contains enough score-distribution
signal to anchor TCM thresholds from positive lower-tail and hard-negative
upper-tail quantiles before running any new TCM continuation.

## Background / Motivation

Earlier threshold sweeps were too broad and made the calibration story hard to
justify. This eval-only run dumps `pos_sim`, `neg_sim_max`, and `neg_top_sim`
on general unseen P31 dev glossaries so threshold choices come from the current
model distribution.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - eval-only, no training
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - ACL6060 disabled
  - dump score distributions for general unseen P31 dev banks
  - TCM losses and hard-negative mining disabled during eval

## Expected metrics

This run is not selected by recall. The expected output is an NPZ dump per
glossary scale plus normal eval logging for traceability.

## Verdict

PENDING: update after both distribution dump jobs finish and the anchored
threshold/weight analysis is generated.

