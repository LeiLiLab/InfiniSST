## Hypothesis

Replacing the BGE-M3 text encoder with BGE-large-en-v1.5 may improve English
domain retrieval for the same varctx576 speech-side setup, especially medicine
and ACL readouts, while preserving dev recall.

## Background / Motivation

The first Taurus 8GPU BGE-large ablation attempt (`ggeqpwie`, Slurm `45236`)
crashed at the step-240 inline eval because full dev + ACL + medicine eval kept
rank 0 busy past the NCCL timeout while the other ranks waited at the barrier.
We still need ACL and medicine inline so the training run has cross-domain W&B
readouts and a secondary checkpoint tracker.  The heavy avoidable part is the
full 12.5k-row dev eval and the over-broad tau diagnostics.

This retry keeps dev, ACL, and medicine inline, but makes the dev pass a
deterministic 100-sample smoke readout at base / gs1000 / gs10000 and reduces
the threshold diagnostics to a single post-filter tau, `0.75`.  Raw
base / gs1000 / gs10000 `recall@10` already covers the no-filter readout.
Eval similarity scoring uses GPU query/text chunking so the expensive MaxSim
matmul runs on the GPU without materializing a full `[N, W, 10000]` tensor.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Baseline anchor: `lh1b88kw` secondary best checkpoint at step `1840`
  (`eval_acl6060/recall@10` tracker), checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Supersedes failed attempt: W&B `ggeqpwie`, Slurm `45236`.
- Diff vs baseline/control:
  - text encoder: `BAAI/bge-m3` -> `BAAI/bge-large-en-v1.5`
  - text encoder preset: `bge-m3` -> `bge-large-en-v1.5`
  - compute: Aries 8GPU control -> Taurus 8GPU ablation
  - batch: control `8192` -> Taurus `8192`
  - GradCache chunk: `128`
  - max steps: `2000` for early ablation readout
  - inline eval interval: `80` -> `240`
  - dev inline eval sample limit: full dev -> deterministic 100 rows (`seed=17`)
  - dev glossary readout: base + `gs1000` + `gs10000`
  - ACL and medicine remain inline full-readout domains
  - ACL/medicine glossary readout: base + `gs10000`
  - eval scoring: CPU post-processing -> GPU chunked scoring
    (`eval_score_device=cuda`, `query_chunk=256`, `text_chunk=1024`)
  - tau diagnostics: `0.85 0.80 0.75 0.70` -> `0.75`
  - qualitative top100 eval dumps: `3` -> `0`
  - ACL gs10k glossary: min-normalized-length-2 backfilled gs10k glossary
  - data, hard-negative depth, audio encoder, MFA windowing, TCM-off setting,
    dev/ACL/medicine eval paths: unchanged

## Expected metrics

Primary metric remains `eval_dev/recall@10_gs10000`, now on the fixed
100-sample dev smoke subset.  Secondary remains `eval_acl6060/recall@10` on the
full inline ACL readout.  Medicine metrics remain full inline cross-domain
readouts.

Use `lh1b88kw` step `1840` secondary-best as the baseline anchor requested for
this ablation, and compare the ablation at matched steps or at both runs'
best-step bundles, explicitly labelling any running runs as not final.  The
100-sample dev metric is only for fast checkpoint tracking; final claims need
a separate full dev/ACL/medicine readout if this ablation looks promising.

## Verdict

Canceled by operator after early readouts showed the BGE-large text-encoder
ablation was unlikely to beat the BGE-M3 control.  The epoch-0 checkpoint is
retained for a standalone full eval-only comparison against the matched
BGE-M3 epoch-0 checkpoint.
