# Full GSV2 k1024 TCM-off dedup resume-to-convergence

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa_gsdedup` / `train`
- **Variant tag**: `hn1024_gsv2full_gsdedup_tcmoff_conv`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_tcmoff_resume3_converge_8gpu_aries.sh`
- **Resume source run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
- **Resume checkpoint**: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep3_bs12k_smallest_dense_normAGGR_8gpu_aries_best.pt`

## Hypothesis

Continuing the deduplicated full GSV2 k1024 TCM-off run from its primary best
checkpoint should show whether the step-1040 dev gs10000 recall plateau is a
true convergence point or a short-horizon artifact. If additional epochs help,
the primary best should refresh before three consecutive lightweight evals pass.

## Background / Motivation

The initial deduplicated run finished successfully on Aries job 45193 and
produced a primary best checkpoint at step 1040, while the final step snapshot
had lower dev gs10000 recall. This continuation keeps the same data, hard
negative depth, TCM-off setting, and dev gs10000 selection metric, then adds an
explicit patience rule so the job stops automatically once the primary metric
stops improving.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
- Diff:
  - checkpoint: resume from the baseline primary best checkpoint at step 1040
  - training horizon: extend from 3 epochs to an epoch-8 scheduler horizon
  - eval cadence: `EVAL_STEPS_SAMPLE=80` -> `50`
  - stop rule: stop after `3` consecutive evals without refreshing `eval_dev/recall@10_gs10000`
  - data and loss: unchanged deduped GSV2 full data, k=1024 per-sample HN, TCM off

## Expected metrics

The continuation is useful if it either improves `eval_dev/recall@10_gs10000`
above the source best `0.9732`, or cleanly stops after three stale evals and
confirms the source run's primary best as the converged checkpoint. Secondary
`eval_dev/topk10_filtered_recall@tau_0p80_gs10000` should not materially
degrade at the new primary best.

## Verdict

PENDING: update after the continuation stops by best-metric patience or reaches
the epoch cap.
