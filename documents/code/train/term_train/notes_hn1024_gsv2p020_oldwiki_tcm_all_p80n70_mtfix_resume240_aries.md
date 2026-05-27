# HN depth + TCM continuation - `k=1024`, GSV2p020 oldwiki, multi-term HN fix

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2p020_oldwiki_mfa` / `train`
- **Variant tag**: `hn1024_p020ow_tcmall_mtfix_r240`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2p020_oldwiki_tcm_all_p80n70_mtfix_resume240_8gpu_aries.sh`
- **Source checkpoint run**: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/d7ij3to1
- **Baseline run candidates**:
  - same `k=1024` HN-depth scout: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
  - same GSV2p020 data line at `k=4096`: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/yx52spnl
  - historical strong `k=512` reference: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/tys70s0y

## Hypothesis

The `d7ij3to1` secondary checkpoint captured useful `k=1024` all-scope TCM state
before the run was manually cancelled. Continuing from that checkpoint with the
new multi-term chunk masking should preserve the useful calibration trajectory
while removing the destructive pressure where co-chunk GT terms could be mined
or trained as negatives under per-sample hard negatives.

## Background / Motivation

The source run used the intended GSV2p020 oldwiki all-scope TCM recipe
(`tcm_pos_threshold=0.80`, `tcm_neg_threshold=0.70`, `hard_neg_k_per_sample=1024`)
and reached a useful secondary checkpoint at step 240. After diagnosing ACL
glossary-expansion failures, the training code was changed so each row carries
all known GT term IDs for the speech chunk; per-sample HN mining excludes every
known co-chunk GT term; and MFA-supervised MaxSim defaults to term-scoped
positives, treating same-chunk different-term rows as neutral.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/fma3wmh2
- Source/resume run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/d7ij3to1
- Resume checkpoint: `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs12k_t=0.07_3var_gsv2p020_oldwiki_gc_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmall_p80n70_bs12k_smallest_dense_normAGGR_8gpu_aries_smoke530_best_acl6060_gs10000.pt`
- Diff:
  - code: enable chunk-positive term IDs, co-chunk GT masking in per-sample HN,
    and MFA term-scoped positives via `mfa_positive_scope=auto`
  - resume global step: checkpoint selected by `best_secondary/step=240`
  - hparams/data/TCM recipe: unchanged from `d7ij3to1`
  - training budget: `epochs=5`, `max_steps=0` (no step cap), `max_train_seconds=0`
    (no Python walltime cap; rely on SLURM only)

## Expected metrics

Compare against `fma3wmh2`, `yx52spnl`, `tys70s0y`, and the source `d7ij3to1` with:

```bash
python documents/code/general/wandb_tool.py --project qwen3_rag compare fma3wmh2 yx52spnl tys70s0y d7ij3to1 <new_run_id> \
  --preset retriever_eval retriever_train --at-best-step --anchor-metric both
```

The run is useful if dev `tau=0.75/0.80` filtered recall and ACL gs10000 recall
do not regress from the source checkpoint while train diagnostics show the new
masking path is active (`pos_count_mean`, `cochunk_neutral_count`,
`hn_false_positive_masked_count`).

## Verdict

PENDING: update after the continuation finishes or shows an early recall/noise failure.
