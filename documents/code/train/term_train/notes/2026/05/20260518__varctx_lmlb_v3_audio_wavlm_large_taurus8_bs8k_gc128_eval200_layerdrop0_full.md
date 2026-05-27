# Retriever Speech Encoder Ablation: WavLM-large Full Training, GC128 LayerDrop0

## Hypothesis

`microsoft/wavlm-large` with BGE-M3 text encoding may provide a faster
speech-side ablation than Whisper-medium.en.  Keeping global batch 8192 and
GradCache chunk size 128 should fit Taurus A6000 memory; disabling WavLM
layerdrop should remove the DDP unused-parameter failure seen in `34sbpz92`
without reducing throughput through a smaller global batch.

## Background / Motivation

Prior WavLM attempts established the required fixes: use the official
`microsoft/wavlm-large` id, patch WavLM `get_input_embeddings()` for PEFT
gradient checkpointing, and keep WavLM-scaled MaxSim windows.  The `gc256`
checkpointing-shim run `hikwfmaa` still OOMed at step 1 backward/recompute.
The `gc128` run `34sbpz92` fit memory but failed because WavLM layerdrop can
skip different LoRA-bearing encoder layers across DDP ranks during GradCache.
This successor keeps `gc128` and disables WavLM `config.layerdrop` in code.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Slow predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/r1pxeaxj
- Failed WavLM predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/hikwfmaa
- Failed WavLM gc128 predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/34sbpz92
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `wavlm-large`
  - `audio_model_id`: `microsoft/wavlm-large`
  - WavLM `get_input_embeddings()` shim keeps PEFT gradient checkpointing enabled
  - WavLM `config.layerdrop`: original HF value -> `0.0`
  - `grad_cache_chunk_size`: `256` -> `128`
  - global batch remains `8192`, per-rank batch remains `1024`
  - MaxSim windows remain `8 12 16 20 24 28 32 40 48 64 80 96`, stride `8`
  - inline eval interval remains `200`, aligned with negative-bank refresh `50`
  - dev sampled 100 rows; ACL and medicine remain full inline readouts

## Expected metrics

Operational target: complete step 2 without DDP unused-parameter failure and
report a usable non-refresh step time.  If `gc128` OOMs after layerdrop is
disabled, fallback is `gc64` before reducing global batch.

## Verdict

Failed after W&B init (`r8q9uf8k`) during step 1 with the same DDP
unused-parameter reduction error.  Disabling WavLM layerdrop was necessary but
not sufficient; the WavLM retriever DDP wrapper also needs
`find_unused_parameters=True` under the GradCache multi-forward training path.
