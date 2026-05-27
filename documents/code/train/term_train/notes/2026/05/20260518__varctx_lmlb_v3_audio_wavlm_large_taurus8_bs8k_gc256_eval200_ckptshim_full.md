# Retriever Speech Encoder Ablation: WavLM-large Full Training, Checkpointing Shim

## Hypothesis

Replacing Qwen3-Omni audio with `microsoft/wavlm-large`, while keeping BGE-M3
text encoding and the same varctx576 data, may provide a faster speech-side
representation than Whisper-medium.en while retaining useful terminology recall.
With a WavLM `get_input_embeddings()` shim, PEFT gradient checkpointing should
fit `bs8k/gc256` on Taurus A6000 without reducing global batch.

## Background / Motivation

The Whisper-medium.en ablation `r1pxeaxj` was operational but too slow, with
about 101s per step at global batch 4096.  The first WavLM attempt `45259`
failed before WandB init because `microsoft/wavlm-large-plus` is not a valid
Hugging Face id.  The second `45260` used `microsoft/wavlm-large` but PEFT
checkpointing failed because `WavLMModel` lacks `get_input_embeddings()`.  The
third `45261` disabled audio checkpointing, reached W&B run `9034wae5`, and then
OOMed at step 1 with `gc256`.

This successor keeps `microsoft/wavlm-large`, patches
`get_input_embeddings()` to WavLM's first feature-extractor convolution, and
therefore keeps audio gradient checkpointing enabled.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Slow predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r1pxeaxj
- Failed attempts: `45259` invalid HF id, `45260` PEFT checkpointing setup,
  `45261` W&B run `9034wae5` OOM with no audio checkpointing.
- Baseline anchor: `lh1b88kw` secondary best checkpoint, Qwen3-Omni audio
  encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `wavlm-large`
  - `audio_encoder_type`: `qwen3_omni` -> `wavlm`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `microsoft/wavlm-large`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `microsoft/wavlm-large`
  - WavLM `get_input_embeddings()` shim for PEFT gradient checkpointing
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

Operational target: pass W&B init and complete step 1 without OOM.  If the
checkpointing shim still OOMs at `gc256`, reduce `grad_cache_chunk_size` to
`128` while preserving global batch 8192 before considering a smaller batch.
Quality target is to recover useful dev/ACL recall after the first few evals.

## Verdict

FAILED at step 1: W&B run `hikwfmaa` initialized successfully and used the WavLM
checkpointing shim, but Slurm job `45262` still OOMed during backward/recompute
with `grad_cache_chunk_size=256`.  This is not a valid training run.  Successor
keeps global batch 8192 and reduces `grad_cache_chunk_size` to `128`.
