# No-Term TCM ACL Raw/1k/10k Eval v3

## Hypothesis

The selected no-term TCM finalist should preserve ACL retrieval quality after
tau filtering across the raw ACL glossary, the 1k expanded glossary, and the
10k expanded glossary, while still reducing no-term emissions on dev v3.

## Background / Motivation

The continuation finalist `T_alpha=0.64, T_beta=0.85, pos_w=1, neg_w=4`
improved the dev v3 no-term operating point.  ACL raw/1k/10k filtered recall is
needed to show whether the same operating threshold remains usable for the ACL
per-paper glossary regimes used by SimulEval.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - eval-only; no training
  - checkpoint: `44205` primary `_best.pt`
  - dev JSONL: balanced dev v3, used for no-term noise
  - ACL JSONL: ACL6060 offline eval dataset
  - ACL glossary banks: raw/base, gt-union `gs1000`, gt-union `gs10000`
  - operating tau: `0.75`

## Expected metrics

Report dense recall and `topk10_filtered_recall@tau_0p75` for ACL raw, ACL
`gs1000`, and ACL `gs10000`; report dev v3 no-term noise at the same tau.

## Verdict

Pending eval-only run.
