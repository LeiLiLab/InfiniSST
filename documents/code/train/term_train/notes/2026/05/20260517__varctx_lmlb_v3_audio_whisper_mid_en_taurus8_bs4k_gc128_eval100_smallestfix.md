# Retriever Speech Encoder Ablation: Whisper Medium.en BS4K Smallest-Fix

## Hypothesis

Whisper medium.en can train as the speech-encoder ablation if we keep the optimized MFA `smallest` MaxSim path and reduce per-rank batch from 1024 to 512, cutting full-batch loss state before the gc128 re-forward/backward phase.

## Background / Motivation

The previous `4qehcl59` run reached the Whisper re-forward/backward phase after fixing the 3000-frame input, padded-window crop, global MaxSim OOM, and fallback shape bug. It then OOMed inside Whisper `fc2` LoRA with about 45.6 GiB already allocated. This run changes the memory pressure knob instead of adding another code change.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qft43kd9
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/09p9rojy
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/prft06bl
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7gbny5qt
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/4qehcl59
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - Whisper code fixes: pad/truncate mel to 3000, crop MaxSim pooling to valid hidden length, optimize MFA `smallest` scoring, and fix fallback indexed assignment.
  - `per_gpu_batch`: `1024` -> `512`
  - global batch: `8192` -> `4096`
  - `grad_cache_chunk_size`: `128`
  - `eval_steps_sample`: `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

First check is operational: step 1 should finish and log a train metric. Because global batch is smaller than the Qwen3-Omni control, early convergence should be interpreted as a feasibility/signal check rather than a final fair metric comparison.

## Verdict

FAILED on Slurm 45257 / W&B `jxllzdon` before completing step 1. The run used per-rank batch 512 but still OOMed during `s_emb.backward(...)` in the GradCache phase-3 Whisper backward recomputation, inside Whisper `fc2`/LoRA dropout. The next relaunch should keep BS4K but reduce `grad_cache_chunk_size` from 128 to 64.
