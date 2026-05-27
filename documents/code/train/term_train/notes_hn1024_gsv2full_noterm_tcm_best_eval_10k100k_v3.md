# No-Term TCM Best Checkpoint 10k/100k Eval v3

## Hypothesis

The 10k-only filtered recall/no-term frontier is not discriminative enough for
choosing the final TCM setting.  Re-evaluating each best checkpoint on both
10k and 100k general unseen P31 glossaries should reveal whether the apparent
Pareto ordering holds as the glossary grows.

## Background / Motivation

Threshold scout v3 and weight scout v3 selected two candidate families:

- balanced/default: `T_alpha=0.64`, especially `pos_w=1, neg_w=4`
- aggressive/noise-first: `T_alpha=0.70`, especially `pos_w=1, neg_w=4`

The previous comparison used each run's best step and operating tau but only
logged `gs10000`.  This eval-only sweep loads the saved `_best.pt` checkpoint
for each candidate and evaluates `gs10000` plus `gs100000` using the same
100k glossary file.  The first 10k entries of that file match the existing
10k glossary.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - eval-only; no training
  - input checkpoints are the `_best.pt` files from the threshold/weight scouts
  - dev JSONL: balanced dev v3
  - eval glossary: 100k general unseen P31 glossary
  - eval sizes: `10000 100000`
  - run-specific operating tau from the training scout is preserved
  - ACL6060 and 1M eval disabled

## Expected metrics

Compare dense recall, run-specific filtered recall, no-term noise, avg kept, and
filtered precision at both `gs10000` and `gs100000`.  Prefer a setting whose
100k behavior remains close to its 10k Pareto position.

## Verdict

Eval-only 10k/100k sweep completed.  The larger 100k glossary makes the
difference much clearer than 10k alone.  The default candidate remains
`T_alpha=0.64, pos_w=1, neg_w=4`: it has `gs100000` dense recall `0.9510`,
filtered recall `0.9485`, and no-term noise `3.1439`, a large noise reduction
over `(1,2)` (`5.5171`) with a small recall cost.  The aggressive candidate is
`T_alpha=0.70, pos_w=1, neg_w=4`: it has `gs100000` dense recall `0.9545`,
filtered recall `0.9441`, and the lowest no-term noise `1.0474`.  Use `0.64,
(1,4)` as the default/reviewer-friendly setting and `0.70, (1,4)` as the
noise-first ablation.
