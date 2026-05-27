# Full GSV2 k1024 TCM-off with variable 2.88-5.76s context, v3

- **Family / data / task**: `sst_ood_hardneg` / `3variant_gsv2full_gsdedup_varctx576` / `train`
- **Variant tag**: `hn1024_varctx576_v3_tcmoff_ep6`
- **Launcher**: `documents/code/train/term_train/run_mfa_smallest_dense_hn1024_gsv2full_gsdedup_varctx_lmlb_v3_tcmoff_ep6_8gpu_aries.sh`
- **Data builder**: `documents/code/data_pre/training_terms_for_retriever/run_build_gsv2full_gsdedup_varctx_lmlb2p88_3p84_4p80_5p76_parallel.sh`

## Hypothesis

Training the retriever on a four-way mix of 2.88s, 3.84s, 4.80s, and 5.76s
speech contexts should better match streaming inference with different
look-back settings than a single fixed 1.92s or 3.84s context. The wider
windows may recover terms that need more acoustic context, while the shorter
windows keep the encoder from overfitting only to long chunks.

## Background / Motivation

The prior 1.92s GSV2-dedup run (`ah9u1bao`) is the direct fixed-context
baseline. The current 3.84s run (`dxwrgbln`) tests a single longer context.
This v3 run keeps the same GSV2-dedup, k=1024 hard negative, MaxSim MFA, and
TCM-off setup, but trains and evaluates on a balanced variable-context dataset
whose durations include the 1.92s inference look-back.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
  - Secondary same-family context baseline URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/dxwrgbln
- Diff:
  - train data: fixed 1.92s / fixed 3.84s context -> balanced 2.88s, 3.84s, 4.80s, 5.76s context
  - eval data: dev and ACL6060 are rebuilt into the same four-duration new_version datasets
  - audio padding: train and eval use `fixed_audio_seconds=5.76`
  - selection: primary best metric is `eval_dev/recall@10_gs10000`; secondary best metric is `eval_acl6060/recall@10`
  - max epochs: 6
  - efficiency probe: smoke tests try `grad_cache_chunk_size=512` first and shrink on OOM before the full run
  - batch fallback: `batch_size=12288` OOMed for all tested GradCache chunks (`512`, `384`, `256`, `128`) with 5.76s fixed audio; the submitted runnable configuration uses `PER_GPU_BATCH=1024`, global `batch_size=8192`, `grad_cache_chunk_size=128`

## Expected metrics

The run should be competitive with the 1.92s baseline on dev `recall@10_gs10000`
and should not degrade ACL6060 base-bank `recall@10`. The main risk is that
5.76s padding increases per-step cost and that wider chunks introduce more
ambiguous terms per speech segment.

## Verdict

PENDING: update after training finishes and compare best-step dev and ACL6060
bundles from WandB using `wandb_tool.py compare --at-best-step`.
