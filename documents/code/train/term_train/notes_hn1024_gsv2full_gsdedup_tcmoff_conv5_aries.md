# Full GSV2 k1024 TCM-off dedup resume-to-convergence, patience 5

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsfix_mfa_gsdedup` / `train`
- **Variant tag**: `hn1024_gsv2full_gsdedup_tcmoff_conv5`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_tcmoff_resume3_converge_8gpu_aries.sh`
- **Resume source run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ig2mjmil
- **Original dedup run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
- **Resume checkpoint**: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2full_gsfix_mfa_gsdedup_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_conv_bs12k_smallest_dense_normAGGR_8gpu_aries.pt`

## Hypothesis

Hard-negative bank refreshes can temporarily depress dev gs10000 recall, so a
three-eval patience window is too short for this continuation. Allowing five
additional stale evals should distinguish a real plateau from post-refresh
recovery while still bounding the run.

## Background / Motivation

The first continuation from the deduplicated best checkpoint stopped at step
1150 after three non-improving evals, but the sequence was still recovering:
`0.9661 -> 0.9717 -> 0.9727` against the restored best `0.9732`. This run
continues from that final checkpoint and gives the primary metric a longer
window to refresh.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ig2mjmil
- Diff:
  - checkpoint: resume from job 45194 final save at step 1150
  - stop rule: `EARLY_STOP_BEST_PATIENCE_EVALS=3` -> `5`
  - naming: use `conv5` variant/run/save names to keep this rerun distinct
  - data, loss, eval glossary, hard-negative depth, and eval cadence: unchanged

## Expected metrics

The run is useful if `eval_dev/recall@10_gs10000` refreshes above `0.9732`
after the hard-negative refresh recovery period, or if five additional stale
evals confirm that the deduplicated source best remains the right checkpoint.
Secondary filtered recall at tau 0.80 should remain close to the source best.

## Verdict

PENDING: update after the patience-5 continuation stops.
