# No-Term TCM Weight Scout v3

## Hypothesis

After threshold scout v3, `T_alpha=0.64` should be the default balanced anchor,
while `T_alpha=0.70` is a useful aggressive noise-suppression anchor.  Sweeping
positive/negative TCM branch weights at these two anchors should reveal whether
extra negative pressure can reduce no-term emissions without materially hurting
dev v3 filtered recall.

## Background / Motivation

Threshold scout v3 selected:

- `T_alpha=0.64`, tau `0.75`: filtered recall `0.9727`, no-term noise `1.52`.
- `T_alpha=0.70`, tau `0.78`: filtered recall `0.9548`, no-term noise `0.46`.

The already-completed threshold scout runs cover `(pos_w=1, neg_w=2)` for both
anchors, so this weight scout only launches the remaining `(1,1)`, `(1,4)`, and
`(2,4)` points.  Final comparison should include the two existing `(1,2)` runs.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Threshold reference runs:
  - `T_alpha=0.64, w=(1,2)`: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/iie3967j
  - `T_alpha=0.70, w=(1,2)`: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/2p0n1i0j
- Diff:
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - dev JSONL: balanced dev v3 (`1:1` with-term/no-term; `1:1` wiki/GigaSpeech positives)
  - fixed `T_beta=0.85`
  - sweep weights `(1,1)`, `(1,4)`, `(2,4)` for `T_alpha in {0.64, 0.70}`
  - each run evaluates only at its own rounded midpoint tau
  - ACL6060 and automatic 1M eval disabled for this scout

## Expected metrics

Select by Pareto tradeoff on dev v3: preserve dense `eval_dev/recall@10_gs10000`,
maximize run-specific filtered recall, and minimize
`eval_dev/noterm_noise@top10_tau_*_gs10000`.  Prefer the `0.64` anchor unless the
`0.70` anchor gives substantially better no-term suppression with acceptable
filtered-recall loss.

## Verdict

Weight scout completed. The best default candidate is `T_alpha=0.64,
pos_w=1, neg_w=4`: at its best step it keeps dense recall high (`0.9752`),
keeps filtered recall acceptable (`0.9661` at tau `0.75`), and cuts no-term
noise to `0.8272` with better filtered precision (`0.1706`). The aggressive
candidate is `T_alpha=0.70, pos_w=1, neg_w=4`: it gives the lowest no-term
noise (`0.3657`) and highest precision (`0.3017`) but pays a larger filtered
recall cost (`0.9529` at tau `0.78`). Increasing both branches to `(2,4)` did
not improve the Pareto frontier.
r.
