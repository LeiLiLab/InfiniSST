# No-Term Anchored TCM Calibration Eval

## Hypothesis

The TCM-off step-2650 baseline can provide a no-term score frontier that fixes
the inference threshold interval before any new TCM training.

## Background / Motivation

The current TCM tuning goal is to reduce false glossary emissions on no-term
speech chunks while preserving good term-bearing recall. We enrich the dev set
with GigaSpeech MFA no-term chunks so low-noise operating points are measured
with a larger no-term sample than the original dev split.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/us4obwe3
- Diff:
  - eval-only, no training
  - resume checkpoint: exported step-2650 TCM-off best checkpoint
  - dev JSONL: original dev plus GigaSpeech no-term add-on
  - ACL6060 disabled
  - dump term-bearing score/rank arrays and no-term raw top10 score arrays

## Expected metrics

Use the raw dump to derive `tau_down` and `tau_center` from the no-term noise
frontier and filtered recall, not from a training sweep.

## Verdict

PENDING: update after calibration dump and frontier analysis finish.
