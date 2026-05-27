# Retriever Speech Encoder Ablation: Whisper Medium.en Smallest-Fix GC128

## Hypothesis

`openai/whisper-medium.en` can run as the speech-encoder ablation at the original per-rank batch size once MFA `smallest` MaxSim scoring avoids full `[B, W, N]` tensors and handles fallback rows with indexed assignment.

## Background / Motivation

This keeps the ablation axis fixed: BGE-M3 text encoder, varctx576 data, Taurus 8 GPU, GradCache chunk 128, and Whisper medium.en speech encoder. The previous `7gbny5qt` run confirmed that the low-memory `smallest` path got past the earlier 26.25 GiB OOM, but failed on a shape bug when only a subset of rows needed fallback.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qft43kd9
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/09p9rojy
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/prft06bl
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/7gbny5qt
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `openai/whisper-medium.en`
  - Whisper code fix 1: pad/truncate mel features to 3000 and preserve valid-length masks after encoder downsampling.
  - Whisper code fix 2: crop `_multiscale_pool` to the maximum valid hidden length before window generation.
  - MaxSim code fix 3: for MFA `smallest`, select the deterministic covering window first, then compute `[B, D] x [N, D]` and per-sample `[B, D] x [B, K, D]` scores.
  - MaxSim code fix 4: fallback rows in optimized `smallest` use indexed assignment, not `torch.where` against a fallback subset.
  - `grad_cache_chunk_size`: `128`
  - `eval_steps_sample`: `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

First check is operational: step 1 should pass global in-batch MaxSim, per-sample hard-negative MaxSim, and the re-forward/backward phase without OOM or shape errors. If it trains, compare early dev/ACL/medicine readouts against the Qwen3-Omni/BGE-M3 control and watch step time.

## Verdict

FAILED on Slurm 45256 / W&B `4qehcl59` before completing step 1. The low-memory `smallest` loss path and fallback fix were both past their previous failure points, but the re-forward/backward phase OOMed inside Whisper `fc2` LoRA with `grad_cache_chunk_size=128`: the process had about 45.6 GiB allocated and tried to allocate another 1.46 GiB. Next relaunch should reduce per-rank batch from 1024 to 512 so full-batch embedding/loss state is smaller before the gc128 re-forward.
