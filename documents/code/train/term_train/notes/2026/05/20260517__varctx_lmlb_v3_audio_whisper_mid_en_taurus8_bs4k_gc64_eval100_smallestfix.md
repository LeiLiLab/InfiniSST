# Retriever Speech Encoder Ablation: Whisper Medium.en BS4K GC64

## Hypothesis

Whisper medium.en can pass the first training step if the optimized MFA `smallest` path is kept, per-rank batch stays at 512, and GradCache re-forward chunks are reduced from 128 to 64.

## Background / Motivation

The previous `jxllzdon` run used BS4K but still OOMed during `s_emb.backward(...)` in the GradCache phase-3 Whisper backward recomputation. The stack pointed to Whisper `fc2` / LoRA dropout, so this run targets activation memory by lowering `grad_cache_chunk_size`.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qft43kd9
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/09p9rojy
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/prft06bl
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7gbny5qt
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/4qehcl59
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/jxllzdon
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - Whisper code fixes: pad/truncate mel to 3000, crop MaxSim pooling to valid hidden length, optimize MFA `smallest` scoring, and fix fallback indexed assignment.
  - `per_gpu_batch`: `512`
  - global batch: `4096`
  - `grad_cache_chunk_size`: `128` -> `64`
  - `eval_steps_sample`: `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

First check is operational: step 1 should finish and log a train metric. If it still OOMs in phase-3 backward, the next viable knobs are `grad_cache_chunk_size=32` or reducing per-rank batch further.

## Verdict

FAILED: canceled by user on 2026-05-18 because throughput was too slow for a
full training run.  The run reached W&B `r1pxeaxj` and passed the first step and
step-100 inline eval, but steady-state training was about 101s/step with
per-rank batch 512, implying roughly 50 hours per epoch.  Treat this as a
valid operational speed/early-readout failure, not an OOM or code crash.
