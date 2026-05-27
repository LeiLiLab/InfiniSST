# No-Term TCM Finalists v3

## Hypothesis

Continuing the converged TCM-off step-2650 baseline with the selected
distribution-anchored TCM setting should reduce no-term emissions under larger
glossaries while preserving dense recall.  The `neg_w=8` run tests whether
stronger negative pressure improves 100k/1M no-term behavior beyond the selected
`neg_w=4` default.

## Background / Motivation

Threshold and weight scouts on balanced dev v3 selected `T_alpha=0.64`,
`T_beta=0.85`, `pos_w=1`, `neg_w=4` as the default/reviewer-friendly setting.
The 10k/100k eval-only sweep confirmed that this setting reduces 100k no-term
noise substantially while keeping filtered recall acceptable.  A higher
`neg_w=8` has not yet been tested at the selected threshold.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Supporting scout/eval runs:
  - selected weight scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/aamk3dok
  - 10k/100k selected eval: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/z2jjzw4p
- Diff:
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - continuation length: +2000 steps (`max_steps=4650`)
  - dev JSONL: balanced dev v3
  - light eval: `gs10000` every 50 steps
  - sparse full eval: `gs100000` every 10 light evals
  - primary best: `eval_dev/recall@10_gs10000`
  - secondary best: `eval_dev_full/recall@10_gs100000`
  - finalists:
    - Aries 8GPU: `T_alpha=0.64`, `pos_w=1`, `neg_w=4`
    - Taurus 7GPU: `T_alpha=0.64`, `pos_w=1`, `neg_w=8`

## Expected metrics

The selected `neg_w=4` finalist should preserve 10k dense recall and improve
100k no-term noise versus the TCM-off baseline.  The `neg_w=8` finalist is
expected to reduce noise further, but may trade off filtered recall; keep it only
if 100k and later 1M confirmation remain acceptable.

## Verdict

PENDING: update after finalist training and 10k/100k/1M confirmation finish.
