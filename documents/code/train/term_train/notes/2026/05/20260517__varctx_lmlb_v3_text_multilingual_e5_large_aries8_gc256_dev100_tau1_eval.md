## Hypothesis

Replacing the BGE-large-en-v1.5 text encoder with `intfloat/multilingual-e5-large`
in retrieval-prefix mode may improve term matching for mixed-domain glossary
strings while keeping the same varctx576 audio/data/training setup.

## Background / Motivation

The Taurus BGE-large-en-v1.5 ablation is still running and appears weak from
early readouts, so this run tests a different large text encoder on Aries while
the Slurm queue is blocked.  E5 requires a retrieval prefix; this launcher uses
`query: ` for all glossary-term text inputs and `mean` text pooling.

The launcher is direct-run friendly: it keeps SBATCH metadata for reproducible
resubmission, but can be executed with `nohup env CUDA_VISIBLE_DEVICES=... bash`
from the Aries checkout.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Baseline anchor: `lh1b88kw` secondary best checkpoint at step `1840`
  (`eval_acl6060/recall@10` tracker), checkpoint:
  `/mnt/gemini/home/jiaxuanluo/train_outputs/q3rag_scale_lora-r128-tr128_bs8k_t=0.07_3var_gsv2full_gsdedup_varctx576_bs8k_gc128_wr1000k_m0.0_maxsim_mfa_variantE_hn1024_tcmoff_ep6_v3_smallest_dense_normAGGR_8gpu_aries_best_eval_acl6060_recallat10.pt`
- Related sibling run: `mhukv2bi`, Slurm `45238`, Taurus BGE-large-en-v1.5
  run with the same fast eval policy.  It is a running sibling, not a final
  metric baseline.
- Diff vs BGE-large sibling:
  - text encoder preset: `bge-large-en-v1.5` -> `multilingual-e5-large`
  - text model id: `BAAI/bge-large-en-v1.5` -> `intfloat/multilingual-e5-large`
  - text input prefix: empty -> `query: `
  - text pooling: `cls` -> `mean`
  - compute: Taurus 8GPU -> Aries 8GPU
  - GradCache chunk remains `256`
- Diff vs baseline/control:
  - data, hard-negative depth, audio encoder, MFA windowing, TCM-off setting,
    fixed/eval audio seconds, dev/ACL/medicine eval paths: unchanged
  - max steps: `2000` for early ablation readout
  - intended inline eval interval: `100`, aligned with negative bank refresh
    cadence (`50` steps); actual failed direct run inherited/used `240`, which
    is not aligned.
  - dev inline eval sample limit: deterministic 100 rows (`seed=17`)
  - dev glossary readout: base + `gs1000` + `gs10000`
  - ACL and medicine remain inline full-readout domains
  - ACL/medicine glossary readout: base + `gs10000`
  - eval scoring: GPU chunked scoring (`eval_score_device=cuda`,
    `query_chunk=256`, `text_chunk=1024`)
  - tau diagnostics: `0.75` only
  - qualitative top100 eval dumps: disabled
  - ACL gs10k glossary: min-normalized-length-2 backfilled gs10k glossary

## Expected metrics

Primary metric remains `eval_dev/recall@10_gs10000` on the fixed 100-sample dev
smoke subset.  Secondary remains `eval_acl6060/recall@10` on the full inline
ACL readout.  Medicine metrics remain full inline cross-domain readouts.

If E5's prefix-aware mean-pooled text space is a better fit than BGE-large for
the glossary-term side, it should recover early dev/ACL signal without changing
the audio or data preparation path.  If it underperforms BGE-large, the text
encoder family is likely not the main bottleneck for this ablation.

## Verdict

FAILED: direct-run process `2970437` / W&B run `om6fnv90` was terminated by
`SIGHUP` at 2026-05-17 22:53:56 UTC after reaching about epoch 0 step 167.
No Python exception, CUDA OOM, or E5 model-loading failure was observed.  The
most likely cause is terminal/session hangup propagation to `torchrun`; relaunch
with `setsid` or inside a real Slurm allocation/hold shell.  Actual run config
used `eval_steps_sample=240`, not the intended 100, likely from inherited env or
an older launcher state.
