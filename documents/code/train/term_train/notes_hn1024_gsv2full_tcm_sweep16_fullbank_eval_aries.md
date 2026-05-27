# Full GSV2 k1024 TCM sweep fullbank dev eval

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa` / `eval`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcm_sweep16_fullbank_eval_1gpu_aries.sh`
- **Training sweep**: `hn1024_gsv2full_tcm_pair_w`

## Hypothesis

A one-shot dev fullbank evaluation with a 1M untrained P31 glossary will expose
settings that only look good at gs10000 because the distractor bank is too
small. The best TCM setting should remain competitive under the larger glossary.

## Background / Motivation

Frequent 1M glossary evaluation is too expensive during training, but running
it once per continuation after training gives a more convincing robustness
check than ACL6060 or a domain-specific 10k glossary.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr
- Diff:
  - task: train -> eval-only
  - resume checkpoint: each TCM continuation final checkpoint
  - eval domain: dev only; ACL6060 disabled
  - eval glossary size: gs10000 -> gs1000000
  - training data loading is limited to one row because no training or hard-negative bank is needed

## Expected metrics

Fullbank winners should preserve the same relative ordering as gs10000 on dev
filtered recall, with acceptable no-term noise at the chosen tau. Large
fullbank regressions mark the setting as overfit to small-bank calibration.

## Verdict

PENDING: update after the fullbank eval array finishes.
