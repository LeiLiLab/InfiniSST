## Hypothesis

Increasing GradCache chunk size from 128 to 256 for the Taurus 8GPU
BGE-large-en-v1.5 varctx576 ablation should reduce per-step overhead without
changing the effective global batch, data, model architecture, or eval policy.

## Background / Motivation

The `mhukv2bi` / Slurm `45238` run showed stable A6000 utilization with roughly
7-9GB apparent memory headroom per GPU at `grad_cache_chunk_size=128`.
Because each rank uses per-GPU batch 1024, chunk 128 creates 8 GradCache
sub-batches per step.  Chunk 256 cuts that to 4 sub-batches, which should
reduce refoward/tokenization/loop overhead while staying more conservative
than chunk 512.

This launcher keeps the fast eval policy from the retry: dev is a deterministic
100-sample smoke readout at base / gs1000 / gs10000, ACL and medicine remain
inline full readouts, eval scoring uses GPU chunked scoring, and tau diagnostics
use only `0.75`.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Baseline anchor: `lh1b88kw` secondary best checkpoint at step `1840`
  (`eval_acl6060/recall@10` tracker), checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Related prior attempt: `mhukv2bi`, Slurm `45238`, same BGE-large setup with
  `grad_cache_chunk_size=128`.
- Diff vs `mhukv2bi`:
  - GradCache chunk: `128` -> `256`
  - variant tag: `vctx576_txt_bgel_t8_d100_tau1` -> `vctx576_txt_bgel_t8_d100_tau1_g256`
  - W&B name includes `gc256`
- Diff vs baseline/control:
  - text encoder: `BAAI/bge-m3` -> `BAAI/bge-large-en-v1.5`
  - compute: Aries 8GPU control -> Taurus 8GPU ablation
  - batch: control `8192` -> Taurus `8192`
  - max steps: `2000` for early ablation readout
  - inline eval interval: `80` -> `100`; this stays aligned with the negative
    bank refresh cadence (`50` steps) so eval happens exactly before every
    second refresh boundary.
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

Primary metric remains `eval_dev/recall@10_gs10000` on the fixed 100-sample dev
smoke subset.  Secondary remains `eval_acl6060/recall@10` on the full inline
ACL readout.  Medicine metrics remain full inline cross-domain readouts.

The operational expectation is faster per-step training than the `gc128`
Taurus BGE-large run without a memory failure.  If chunk 256 reaches step 100
and completes dev/ACL/medicine inline eval, treat it as the preferred launcher
default over chunk 128 for this ablation family.

## Verdict

PENDING: fill after the run finishes.
