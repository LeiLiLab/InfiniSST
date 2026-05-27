# No-Term TCM Threshold Scout v3

## Hypothesis

With `T_beta=0.85` fixed at the highest baseline operating point that preserves
about `0.90` filtered recall, a small negative-threshold scout can identify the
minimum useful negative coverage before sweeping TCM weights.

## Background / Motivation

The repaired dev v2/v3 evaluation is glossary-conditioned: no-term chunks are
removed or relabelled when they contain active glossary terms.  The baseline
frontier shows `T_beta=0.85` is explainable from filtered recall, while
candidate `T_alpha` anchors are:

- `0.60`: no-term `avg_emitted` first becomes smaller than `10.0`.
- `0.64`: no-term `pass_rate` first becomes smaller than `1.0`.
- `0.70`/`0.72`: stress points on the steep noise descent, used to test whether
  the data-derived anchors are too broad.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - dev JSONL: balanced dev v3 (`1:1` with-term/no-term; `1:1` wiki/GigaSpeech positives)
  - fixed `T_beta=0.85`, `pos_weight=1`, `neg_weight=2`
  - sweep `T_alpha in {0.60, 0.64, 0.70, 0.72}`
  - each run evaluates only at its own `inference_tau=round((T_beta+T_alpha)/2, 0.01)`
  - ACL6060 and automatic 1M eval disabled for the scout

## Expected metrics

Select a threshold by Pareto tradeoff: preserve dev v3 filtered recall at the
run-specific inference tau while reducing no-term emitted candidates.  If
`T_alpha=0.64` is close to the slope-entry stress points, prefer it because it
has the cleanest data-derived interpretation; otherwise use the stress point
that materially improves no-term noise without collapsing recall.

## Verdict

Threshold scout completed.  `T_alpha=0.64` is the best first weight-scout
anchor: it preserves filtered recall (`0.9727` at tau `0.75`) while reducing
no-term noise to `1.52`.  `T_alpha=0.70/0.72` are useful stress anchors because
they reduce no-term noise much further (`0.46/0.40`) but pay a larger filtered
recall cost (`0.9548/0.9504`).
