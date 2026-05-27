# Full GSV2 k1024 TCM-off resume-to-convergence

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa` / `train`
- **Variant tag**: `hn1024_gsv2full_tcmoff_r3fixeval`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_tcmoff_resume3_converge_8gpu_aries.sh`
- **Resume source run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/k6e9askw
- **Full GSV2 baseline run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ly6sc2mr

## Hypothesis

Before setting TCM thresholds, the TCM-off retriever should be trained closer to
convergence and inspected under a general unseen wiki glossary. A converged
TCM-off baseline should provide a more stable score distribution and best dev
recall checkpoint for distribution-guided TCM calibration.

## Background / Motivation

The first 16-way TCM pair/weight sweep did not clearly improve the appropriate
per-setting filtered recall. This suggests the next step should not be a wider
TCM grid, but a stronger TCM-off baseline plus score-distribution analysis.
The dev gs10000 bank here uses P31-ranked unseen wiki terms, not the
domain-specific CS/NLP/AI glossary.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/k6e9askw
- Diff:
  - resume checkpoint: best dev gs10000 checkpoint from `us4obwe3` at step 2650
  - hparam `tcm_pos_loss_weight` / `tcm_neg_loss_weight`: `0.0` / `0.0` unchanged
  - hparam `epochs`: `3` -> `8`
  - hparam `scheduler_epochs`: `4` -> `8`
  - scheduler: restore model/optimizer momentum and scheduler state from the best checkpoint
  - eval selection: best checkpoint metric remains `eval_dev/recall@10_gs10000`
  - lightweight eval: general unseen P31 10k glossary every 50 optimizer steps
  - sparse 100k eval: general unseen P31 100k glossary every 10 lightweight evals
  - secondary best checkpoint: `eval_dev_full/recall@10_gs100000`
  - 1M full eval: auto-submitted as a separate Taurus eval-only SLURM job whenever the primary 10k best checkpoint is refreshed

## Expected metrics

The run should either improve dev `recall@10_gs10000` beyond the 3-epoch
baseline or reveal that the TCM-off model has already saturated. The resulting
best checkpoint will be used for score-distribution dumps and threshold-guided
TCM experiments.

## Verdict

PENDING: update after convergence run finishes and best dev gs10000 checkpoint
is selected.
