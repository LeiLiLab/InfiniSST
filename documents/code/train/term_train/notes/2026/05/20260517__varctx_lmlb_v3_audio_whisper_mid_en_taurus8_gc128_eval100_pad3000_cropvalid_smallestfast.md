# Retriever Speech Encoder Ablation: Whisper Medium.en Smallest-Fast GC128

## Hypothesis

`openai/whisper-medium.en` can be tested as a speech-encoder replacement for Qwen3-Omni at the original per-rank batch size if MFA `smallest` MaxSim scoring selects the deterministic covering window before the text matmul instead of materializing the full `[B, W, N]` tensor.

## Background / Motivation

This keeps the ablation axis fixed: BGE-M3 text encoder, varctx576 data, Taurus 8 GPU, GradCache chunk 128, and Whisper medium.en speech encoder. The previous `prft06bl` run reached W&B init and step 1 but OOMed in `_maxsim_score_mfa` because the loss path still built full-window global in-batch similarities even though `mfa_window_selection=smallest` only needs one selected window per row.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qft43kd9
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/09p9rojy
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/prft06bl
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `openai/whisper-medium.en`
  - Whisper code fix 1: pad/truncate mel features to 3000 and preserve valid-length masks after encoder downsampling.
  - Whisper code fix 2: crop `_multiscale_pool` to the maximum valid hidden length before window generation.
  - MaxSim code fix 3: for MFA `smallest`, select the deterministic covering window first, then compute `[B, D] x [N, D]` and per-sample `[B, D] x [B, K, D]` scores, avoiding full `[B, W, N]` and `[B, W, K]` tensors.
  - `grad_cache_chunk_size`: `128`
  - `eval_steps_sample`: `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

First check is operational: step 1 should pass the global in-batch MaxSim loss and per-sample hard-negative loss without OOM. If it trains, compare early dev/ACL/medicine readouts against the Qwen3-Omni/BGE-M3 control and watch step time.

## Verdict

FAILED on Slurm 45255 / W&B `7gbny5qt` before completing step 1. The optimized `smallest` path avoided the previous 26.25 GiB OOM, but a fallback-row shape bug used `torch.where(needs_fallback.unsqueeze(1), fallback_sim, sim_out)` where `fallback_sim` only contains the fallback subset. Fix by assigning `sim_out[needs_fallback] = fallback_sim` in both global and per-sample MFA `smallest` paths.
