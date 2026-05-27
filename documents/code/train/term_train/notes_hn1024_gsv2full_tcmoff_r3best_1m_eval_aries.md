# Full GSV2 k1024 TCM-off 1M Eval

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa` / `eval`
- **Variant tag**: `hn1024_gsv2full_r3best_eval1m`
- **Launcher**: `documents/code/train/term_train/eval_mfa_smallest_dense_hn1024_gsv2full_tcmoff_r3best_1m_1gpu_aries.sh`
- **Checkpoint source run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/058tdx9a

## Hypothesis

The best dev gs10000 checkpoint from the resumed TCM-off baseline should remain
competitive when evaluated against a 1M general unseen P31 wiki glossary.

## Background / Motivation

Running 1M full eval inside the DDP training loop exceeded the NCCL collective
timeout because rank 0 stayed in evaluation while the remaining ranks waited.
This job evaluates the checkpoint in a separate one-GPU eval-only process.
It is scheduled on Taurus so it does not compete with the Aries 8GPU training job.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/058tdx9a
- Diff:
  - load checkpoint: `058tdx9a` best dev gs10000 checkpoint
  - eval-only: no training, no hard-negative bank, no TCM sweep diagnostics
  - eval glossary: general unseen P31 1M glossary
  - metric: `eval_dev/recall@10_gs1000000`

## Expected metrics

The run should produce a stable 1M fullbank recall number for the baseline
checkpoint without blocking DDP training ranks or triggering NCCL timeout.

## Verdict

SUCCESS: one-shot Taurus 1GPU 1M eval completed and logged to WandB.
