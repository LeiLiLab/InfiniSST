## Hypothesis

Replacing the BGE-M3 text encoder with BGE-large-en-v1.5 may improve English
domain retrieval for the same varctx576 speech-side setup, especially medicine
and ACL readouts, while preserving dev recall.

## Background / Motivation

The current variable-context control uses Qwen3-Omni audio features and BGE-M3
text embeddings.  It is expensive: dev, ACL, and medicine inline evals all run
on each eval cycle, and the current control uses an 80-step eval interval.  This
ablation keeps the varctx576 data and Qwen3-Omni audio side fixed, changes only
the text encoder to `BAAI/bge-large-en-v1.5`, runs on Taurus 8GPU, and reduces
inline eval frequency to every 240 steps to cut eval overhead.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Baseline anchor: `lh1b88kw` secondary best checkpoint at step `1840`
  (`eval_acl6060/recall@10` tracker), checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Supporting baseline/control runs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/ah9u1bao
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/dxwrgbln
- Diff:
  - text encoder: `BAAI/bge-m3` -> `BAAI/bge-large-en-v1.5`
  - text encoder preset: `bge-m3` -> `bge-large-en-v1.5`
  - compute: Aries 8GPU control -> Taurus 8GPU ablation
  - batch: control `8192` -> Taurus `8192`
  - GradCache chunk: `128`
  - max steps: `2000` for early ablation readout
  - inline eval interval: `80` -> `240`
  - qualitative top100 eval dumps: `3` -> `0`
  - ACL gs10k glossary: min-normalized-length-2 backfilled gs10k glossary
  - data, hard-negative depth, audio encoder, MFA windowing, TCM-off setting, dev/ACL/medicine eval paths: unchanged

## Expected metrics

Primary metric remains `eval_dev/recall@10_gs10000`; secondary remains
`eval_acl6060/recall@10`.  Medicine metrics are a cross-domain readout.  Use
`lh1b88kw` step `1840` secondary-best as the baseline anchor requested for this
ablation, and compare the ablation at matched steps or at both runs' best-step
bundles, explicitly labelling any running runs as not final.

## Verdict

FAILED: Slurm `45236` / W&B `ggeqpwie` crashed after step 240 inline eval.  The
run reached W&B init and training proceeded to step 240, but rank 0 spent longer
than the 7200s NCCL timeout inside dev+ACL+medicine eval while non-main ranks
waited at the post-eval distributed barrier, causing an NCCL `ALLREDUCE`
watchdog timeout and torchrun `SIGABRT`.
