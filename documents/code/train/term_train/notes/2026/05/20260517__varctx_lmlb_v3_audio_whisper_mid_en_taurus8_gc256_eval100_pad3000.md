# Retriever Speech Encoder Ablation: Whisper Medium.en Pad3000 Relaunch

## Hypothesis

Replacing the default Qwen3-Omni audio encoder with `openai/whisper-medium.en`, while keeping the BGE-M3 text encoder and varctx576 retriever training data, may improve or degrade speech-side alignment in a way that is worth measuring separately from text-encoder changes.

## Background / Motivation

The first Whisper-mid attempt reached W&B as run `o0p64pr7` but failed before completing step 1 because Whisper requires fixed 3000-frame mel features and the 5.76s varctx collate path emitted 576-frame mel features. The train script now pads/truncates Whisper mel features to 3000 before calling the encoder and maps the original mel length through the encoder downsampling ratio for MaxSim pooling masks.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `openai/whisper-medium.en`
  - `lora_target_modules`: Qwen targets including `proj1/proj2` -> Whisper targets `q_proj k_proj v_proj out_proj fc1 fc2`
  - Whisper-specific code fix: pad/truncate mel features to 3000 and preserve valid-length masks after encoder downsampling.
  - `grad_cache_chunk_size`: control `128` -> ablation `256`
  - `eval_steps_sample`: control `80` -> ablation `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

The first checkpoint should be compared against the Qwen3-Omni/BGE-M3 control and the BGE-M3 epoch-0 eval-only control. A useful result would retain dev `recall@10_gs10000` within roughly 1 pp of the control while improving ACL/medicine readouts or reducing step time.

## Verdict

FAILED on Slurm 45252 / W&B `qft43kd9` before completing step 1. The Whisper pad3000 fix passed the input-length check, but `grad_cache_chunk_size=256` OOMed on 48GB A6000s because Whisper's fixed 3000-frame encoder input raised activation memory to about 42.1 GiB before an additional 5.86 GiB allocation. Relaunch with the same scientific setup but `grad_cache_chunk_size=128`.
