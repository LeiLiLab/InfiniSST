# Retriever Speech Encoder Ablation: WavLM-large-plus Full Training

## Hypothesis

Replacing the default Qwen3-Omni audio encoder with `microsoft/wavlm-large-plus`,
while keeping BGE-M3 text encoding and the same varctx576 data, may provide a
better speech-side representation than Whisper-medium.en without the fixed
3000-frame Whisper bottleneck.

## Background / Motivation

The Whisper-medium.en ablation `r1pxeaxj` passed step 1 and produced a step-100
readout, but its steady-state step time was about 101s with global batch 4096,
implying roughly 50 hours per epoch.  This is too slow for the current paper
timeline.  WavLM consumes raw waveform features and should avoid Whisper's
padding-heavy encoder path.

This launcher also fixes the MFA time mapping used by MaxSim supervision:
window start/end times are now computed from the actual fixed context duration
and inferred encoder length.  For WavLM's higher frame rate, the launcher uses
4x larger frame windows and stride so the effective time spans match the
Qwen3-Omni retriever while keeping the number of MaxSim windows small.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Slow predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r1pxeaxj
- Baseline anchor: `lh1b88kw` secondary best checkpoint, Qwen3-Omni audio
  encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `wavlm-large-plus`
  - `audio_encoder_type`: `qwen3_omni` -> `wavlm`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `microsoft/wavlm-large-plus`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `microsoft/wavlm-large-plus`
  - audio LoRA targets: `q_proj k_proj v_proj out_proj intermediate_dense output_dense`
  - MaxSim windows: `2 3 4 5 6 7 8 10 12 16 20 24` -> `8 12 16 20 24 28 32 40 48 64 80 96`
  - MaxSim stride: `2` -> `8`
  - dynamic MFA window time mapping in `qwen3_glossary_neg_train.py`
  - global batch: `8192`, per-rank batch `1024`
  - `grad_cache_chunk_size`: `256`
  - full training: `MAX_STEPS=0`, `EPOCHS=6`
  - inline eval interval: `200`, aligned with negative-bank refresh `50`
  - dev inline eval sample limit: deterministic 100 rows (`seed=17`)
  - ACL and medicine remain full inline readouts
  - tau diagnostics: `0.75` only

## Expected metrics

First operational target: step time should be far below the Whisper-medium.en
run and should make an epoch feasible on Taurus.  Quality target is to recover
useful dev/ACL recall after the first few evals.  If WavLM is still too slow or
weak, the next knobs are reducing full inline eval frequency further or
returning to Qwen3-Omni audio with text/data-side changes.

## Verdict

FAILED before WandB init: Slurm job `45259` exited after model loading because
`microsoft/wavlm-large-plus` is not a valid Hugging Face model id.  This attempt
is invalid as a training experiment; successor uses the official
`microsoft/wavlm-large` repository.
