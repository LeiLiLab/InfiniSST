# Retriever Speech Encoder Ablation: Whisper Medium.en Pad3000 GC128

## Hypothesis

`openai/whisper-medium.en` can be evaluated as a speech-encoder replacement for Qwen3-Omni if we adapt its fixed 3000-frame mel input and reduce GradCache chunk size enough to fit on 48GB A6000s.

## Background / Motivation

The BGE-large text-encoder ablation underperformed BGE-M3, so this run keeps BGE-M3 fixed and moves the ablation axis to the speech encoder. Two predecessor runs established necessary implementation constraints: `o0p64pr7` failed because Whisper requires 3000 mel frames, and `qft43kd9` passed that check but OOMed with `grad_cache_chunk_size=256`.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qft43kd9
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `openai/whisper-medium.en`
  - `lora_target_modules`: Qwen targets including `proj1/proj2` -> Whisper targets `q_proj k_proj v_proj out_proj fc1 fc2`
  - Whisper-specific code fix: pad/truncate mel features to 3000 and preserve valid-length masks after encoder downsampling.
  - `grad_cache_chunk_size`: control `128`, failed Whisper run `256`, this relaunch `128`
  - `eval_steps_sample`: control `80` -> ablation `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

First check is operational: step 1 should pass without the Whisper length error or gc256 OOM. If it trains, compare early dev/ACL/medicine readouts against the Qwen3-Omni/BGE-M3 control and watch step time, since Whisper's forced 3000-frame input may be slower than Qwen3-Omni for 5.76s audio.

## Verdict

FAILED on Slurm 45253 / W&B `09p9rojy` before completing step 1. The run passed Whisper length adaptation and `gc128` encoder forward, but OOMed when concatenating full-batch MaxSim speech chunks because padded Whisper hidden length 1500 caused `_multiscale_pool` to produce many invalid padded-tail windows. Follow-up fix crops MaxSim pooling to the maximum valid hidden length before window generation.
