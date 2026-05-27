# TCM-Off Step-2650 ACL Raw/1k/10k Eval v3

## Hypothesis

The TCM-off step-2650 baseline should provide the necessary control for judging
whether the selected no-term TCM finalist actually improves tau-filtered ACL and
dev-v3 operating behavior at the same threshold.

## Background / Motivation

The `final_n64_p1n4` result was evaluated at `tau=0.75` on dev v3 and ACL
raw/gs1k/gs10k, but those numbers are hard to interpret without the converged
TCM-off checkpoint that seeded the TCM continuation. This eval reuses the same
dataset, glossary, and threshold settings as the finalist eval.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - eval-only; no training
  - checkpoint: TCM-off `us4obwe3` exported primary `_best.pt`
  - TCM loss weights remain disabled: `pos_w=0`, `neg_w=0`
  - dev JSONL: balanced dev v3, used for no-term noise
  - ACL JSONL: ACL6060 offline eval dataset
  - ACL glossary banks: raw/base, gt-union `gs1000`, gt-union `gs10000`
  - operating tau: `0.75`

## Expected metrics

Report dense recall and `topk10_filtered_recall@tau_0p75` for ACL raw, ACL
`gs1000`, and ACL `gs10000`; report dev v3 no-term noise at the same tau so the
finalist can be compared against the TCM-off control.

## Verdict

Completed eval-only TCM-off control for the final_n64_p1n4 ACL/dev-v3 comparison.
