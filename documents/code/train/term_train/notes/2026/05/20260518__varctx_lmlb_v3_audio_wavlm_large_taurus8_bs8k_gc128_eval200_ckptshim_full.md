# Retriever Speech Encoder Ablation: WavLM-large Full Training, GC128

## Hypothesis

`microsoft/wavlm-large` with BGE-M3 text encoding may provide a faster
speech-side ablation than Whisper-medium.en.  Keeping global batch 8192 while
reducing GradCache chunk size from 256 to 128 should fit Taurus A6000 memory
without increasing epoch count.

## Background / Motivation

Prior WavLM attempts established the required fixes: use the official
`microsoft/wavlm-large` id, patch WavLM `get_input_embeddings()` for PEFT
gradient checkpointing, and keep WavLM-scaled MaxSim windows.  The `gc256`
checkpointing-shim run `hikwfmaa` still OOMed at step 1 backward/recompute, so
the next conservative memory fix is `grad_cache_chunk_size=128` while preserving
global batch 8192.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Slow predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r1pxeaxj
- Failed WavLM predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/hikwfmaa
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `wavlm-large`
  - `audio_model_id`: `microsoft/wavlm-large`
  - WavLM `get_input_embeddings()` shim keeps PEFT gradient checkpointing enabled
  - `grad_cache_chunk_size`: `256` -> `128`
  - global batch remains `8192`, per-rank batch remains `1024`
  - MaxSim windows remain `8 12 16 20 24 28 32 40 48 64 80 96`, stride `8`
  - inline eval interval remains `200`, aligned with negative-bank refresh `50`
  - dev sampled 100 rows; ACL and medicine remain full inline readouts

## Expected metrics

Operational target: complete step 1 without OOM and report a usable step time.
If `gc128` still OOMs, fallback is `gc64` before reducing global batch.

## Verdict

Failed after W&B init (`34sbpz92`) during step 1 with DDP unused-parameter
reduction errors.  This was not an OOM: the run reached roughly 29GB/GPU and
then failed because WavLM layerdrop can skip different encoder layers across
ranks, leaving LoRA parameters unused in a GradCache DDP iteration.  Successor
keeps `gc128` but disables WavLM `config.layerdrop` during training.
