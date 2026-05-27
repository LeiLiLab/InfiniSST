# GigaSpeech-only HN1024 varctx 8-GPU best-checkpoint resume

## Hypothesis

Resuming the GigaSpeech-only ablation from its primary best checkpoint on all 8
Aries GPUs should continue the same data-ablation test with higher throughput
than the initial 4-GPU run, without changing the training/eval data or the
model-selection rule.

## Background / Motivation

The source ablation run is W&B `g49qabuf`, launched by
`20260525T1248__retriever_train__varctx576_hn1024_gsonly_tcmoff_ep6_aries4`.
It used Aries GPUs 0,1,2,3 and saved a primary best checkpoint at step 320.
The user requested cancelling that 4-GPU task and continuing on Aries using all
8 GPUs from the best checkpoint.

## What changed vs baseline

- Resume checkpoint is the source run primary best:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_gsonly_bs8192_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_4gpu_aries_best.pt`
- Compute changes from Aries GPUs 0,1,2,3 to Aries GPUs 0,1,2,3,4,5,6,7.
- Global batch remains 8192 with per-rank batch 1024 and GradCache chunk 128.
- `save_latest_steps=50` is enabled for pause-safe recovery during the resumed
  run; the eval cadence and best metrics remain aligned with the source run.

## Expected metrics

The resumed run should preserve the source run's selection rule:
primary `eval_dev/recall@10_gs10000`, secondary `eval_acl6060/recall@10`.
ACL remains a held-out readout and is not used for checkpoint or hyperparameter
selection.

## Verdict

STARTUP VERIFIED: launched detached on Aries as PID `3954026` using physical
GPUs 0,1,2,3,4,5,6,7.  The log confirms restore from the requested primary best
checkpoint:
`[RESUME] ..._4gpu_aries_best.pt epoch=1 step=320`.

W&B run: `0rs042wc`
`https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/0rs042wc`
