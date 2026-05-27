# Distribution-Anchored TCM Scout Sweep

## Hypothesis

Starting from the converged TCM-off step-2650 checkpoint, a compact TCM
continuation with thresholds anchored by dev score distributions can improve
threshold consistency without sacrificing 10k general unseen P31 recall.

## Background / Motivation

The previous broad threshold sweep was not aligned with the operating threshold
for each setting. This sweep first fixes one distribution-derived center
threshold pair, then explores asymmetric positive/negative TCM branch weights
chosen from observed violation pressure.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - TCM thresholds and weight pairs are read from `tcm_anchor_scout_grid.tsv`
  - full GSV2 data, per-sample hard negatives `k=1024`
  - short continuation budget with eval every 50 steps
  - primary best metric: `eval_dev/recall@10_gs10000`
  - secondary best metric: `eval_dev_full/recall@10_gs100000`
  - ACL6060 and automatic 1M eval are disabled for scouts

## Expected metrics

Select by aligned best-step 10k recall first, then use 100k secondary and score
distribution pressure to pick a stable finalist for longer confirmation.

## Verdict

PENDING: update after scout runs finish and are compared at exported best steps.

