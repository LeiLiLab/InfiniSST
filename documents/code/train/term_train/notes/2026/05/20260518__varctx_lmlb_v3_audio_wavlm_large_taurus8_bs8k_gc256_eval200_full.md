# Retriever Speech Encoder Ablation: WavLM-large Full Training

## Hypothesis

Replacing the default Qwen3-Omni audio encoder with the official
`microsoft/wavlm-large` model, while keeping BGE-M3 text encoding and the same
varctx576 data, may provide a faster speech-side representation than
Whisper-medium.en while retaining useful terminology recall.

## Background / Motivation

The Whisper-medium.en ablation `r1pxeaxj` passed step 1 and produced a step-100
readout, but its steady-state step time was about 101s with global batch 4096,
implying roughly 50 hours per epoch.  That is too slow for the paper timeline.

The first WavLM attempt `45259` used the nonexistent Hugging Face id
`microsoft/wavlm-large-plus` and failed before WandB init.  This successor uses
the official `microsoft/wavlm-large` repository.  The launcher also uses dynamic
MFA time mapping for MaxSim supervision and WavLM-scaled frame windows/stride so
the effective time spans match the Qwen3-Omni retriever without exploding the
number of MaxSim windows.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Slow predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r1pxeaxj
- Failed WavLM-plus attempt: Slurm job `45259`, no WandB run id.
- Baseline anchor: `lh1b88kw` secondary best checkpoint, Qwen3-Omni audio
  encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `wavlm-large`
  - `audio_encoder_type`: `qwen3_omni` -> `wavlm`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `microsoft/wavlm-large`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `microsoft/wavlm-large`
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
useful dev/ACL recall after the first few evals.  If WavLM-large is still too
slow or weak, the next knobs are reducing full inline eval frequency further,
dropping batch/chunk size only if needed for memory, or returning to Qwen3-Omni
audio with text/data-side changes.

## Verdict

FAILED before WandB init: Slurm job `45260` loaded `microsoft/wavlm-large`, but
PEFT failed while preparing gradient checkpointing because `WavLMModel` does not
implement `get_input_embeddings()`.  This attempt is invalid as a training
experiment; successor disables audio gradient checkpointing for WavLM.
