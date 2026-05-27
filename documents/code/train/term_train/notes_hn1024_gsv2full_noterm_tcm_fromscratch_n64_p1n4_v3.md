# No-Term TCM From-Scratch n64 p1n4 v3

## Hypothesis

Training with the selected TCM operating point from step 0 should learn a more
threshold-stable retriever than continuing from the TCM-off baseline, while
preserving dev-v3 recall on the general unseen 10k glossary.

## Background / Motivation

The distribution-anchored scouts selected `T_alpha=0.64`, `T_beta=0.85`,
`pos_w=1`, `neg_w=4` as the reviewer-friendly default: it reduces no-term noise
on the 100k glossary with a small recall cost.  A from-scratch run tests whether
the same constraint is easier to satisfy when it shapes the representation
throughout training rather than only during a continuation phase.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - train from scratch; no `RESUME`
  - full GSV2 train JSONL with MFA repairs
  - hard negative mining: per-sample `k=1024`
  - TCM from step 0: `T_alpha=0.64`, `T_beta=0.85`, `pos_w=1`, `neg_w=4`
  - batch: `8 * 1536 = 12288` on Aries
  - max steps: `4650`
  - light eval: dev v3 with general unseen 10k glossary every 50 steps
  - secondary eval: general unseen 100k glossary every 10 light evals
  - primary best: `eval_dev/recall@10_gs10000`
  - secondary best: `eval_dev_full/recall@10_gs100000`

## Expected metrics

Match or exceed the continuation finalist on 10k/100k recall while keeping the
tau-filtered no-term noise near the selected `n64_p1n4` operating point.

## Verdict

Pending from-scratch training run.
