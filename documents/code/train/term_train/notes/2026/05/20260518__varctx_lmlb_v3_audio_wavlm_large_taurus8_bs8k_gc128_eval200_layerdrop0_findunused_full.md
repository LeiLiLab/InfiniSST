# Retriever Speech Encoder Ablation: WavLM-large Full Training, GC128 LayerDrop0 FindUnused

## Hypothesis

`microsoft/wavlm-large` with BGE-M3 text encoding may provide a faster
speech-side ablation than Whisper-medium.en.  Keeping global batch 8192 and
GradCache chunk size 128 should fit Taurus A6000 memory; disabling WavLM
layerdrop and enabling DDP unused-parameter detection for the WavLM retriever
should remove the DDP failure seen in `34sbpz92` and `r8q9uf8k` without
reducing throughput through a smaller global batch.

## Background / Motivation

Prior WavLM attempts established the required fixes: use the official
`microsoft/wavlm-large` id, patch WavLM `get_input_embeddings()` for PEFT
gradient checkpointing, and keep WavLM-scaled MaxSim windows.  The `gc256`
checkpointing-shim run `hikwfmaa` still OOMed at step 1 backward/recompute.
The `gc128` run `34sbpz92` fit memory but failed with DDP unused-parameter
errors.  The layerdrop0 retry `r8q9uf8k` showed that disabling WavLM layerdrop
alone is not sufficient for this GradCache path.  This successor keeps `gc128`,
keeps WavLM `config.layerdrop=0.0`, and wraps the WavLM retriever DDP with
`find_unused_parameters=True`.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Slow predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r1pxeaxj
- Failed WavLM predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/hikwfmaa
- Failed WavLM gc128 predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/34sbpz92
- Failed WavLM layerdrop0 predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r8q9uf8k
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `wavlm-large`
  - `audio_model_id`: `microsoft/wavlm-large`
  - WavLM `get_input_embeddings()` shim keeps PEFT gradient checkpointing enabled
  - WavLM `config.layerdrop`: original HF value -> `0.0`
  - WavLM retriever DDP: `find_unused_parameters=True`
  - `grad_cache_chunk_size`: `256` -> `128`
  - global batch remains `8192`, per-rank batch remains `1024`
  - MaxSim windows remain `8 12 16 20 24 28 32 40 48 64 80 96`, stride `8`
  - inline eval interval remains `200`, aligned with negative-bank refresh `50`
  - dev sampled 100 rows; ACL and medicine remain full inline readouts

## Expected metrics

Operational target: complete step 2 without DDP unused-parameter failure and
report a usable non-refresh step time.  If `gc128` OOMs after DDP find-unused
is enabled, fallback is `gc64` before reducing global batch.

## Verdict

Cancelled by user on 2026-05-18 after Slurm job `45265` was handed off for a
rerun on other resources. Treat this W&B run as partial and non-final.
