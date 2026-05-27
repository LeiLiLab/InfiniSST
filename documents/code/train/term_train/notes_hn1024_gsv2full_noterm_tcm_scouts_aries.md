# No-Term Anchored TCM Scout Sweep

## Hypothesis

A small negative-heavy TCM continuation can reduce no-term glossary emissions at
the calibrated operating interval while preserving 10k general unseen P31 recall.

## Background / Motivation

The TCM-off baseline frontier on enriched dev selected `tau_down=0.79` and
`tau_center=0.82`. The corresponding training bracket is `T_alpha=0.77`,
`T_beta=0.84`, leaving room to lower inference tau under OOD/domain shift.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - dev JSONL: original dev plus GigaSpeech no-term add-on
  - TCM thresholds: `T_beta=0.84`, `T_alpha=0.77`
  - scout weights read from `noterm_tcm_scout_grid.tsv`
  - eval sweep thresholds: only `0.79 0.82`
  - ACL6060 and automatic 1M eval disabled for scouts

## Expected metrics

Select by Pareto tradeoff: keep dense `eval_dev/recall@10_gs10000` stable,
preserve filtered recall at `tau_down/tau_center`, and reduce
`noterm_noise@top10_tau_*_gs10000` versus the TCM-off baseline frontier.

## Verdict

PENDING: update after no-term anchored scout runs finish and are compared.
