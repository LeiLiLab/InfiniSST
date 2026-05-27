# Retriever Speech Encoder Ablation: Whisper Medium.en Pad3000 Valid-Crop GC128

## Hypothesis

`openai/whisper-medium.en` can be tested as a speech-encoder replacement for Qwen3-Omni if its fixed 3000-frame mel input is padded/truncated and MaxSim pooling ignores padded hidden frames before building dense windows.

## Background / Motivation

The text side stays fixed at BGE-M3 because the BGE-large text-encoder ablation did not look promising. This run continues the Whisper speech-encoder ablation after three controlled failures: `o0p64pr7` exposed Whisper's fixed-length mel requirement, `qft43kd9` showed `grad_cache_chunk_size=256` is too large on 48GB A6000s, and `09p9rojy` showed padded Whisper hidden frames create too many invalid MaxSim windows unless pooling is cropped to the valid hidden length.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Failed predecessor run URLs:
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/o0p64pr7
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/qft43kd9
  - https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/09p9rojy
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `openai/whisper-medium.en`
  - `lora_target_modules`: Qwen targets including `proj1/proj2` -> Whisper targets `q_proj k_proj v_proj out_proj fc1 fc2`
  - Whisper code fix 1: pad/truncate mel features to 3000 and preserve valid-length masks after encoder downsampling.
  - Whisper code fix 2: crop `_multiscale_pool` to the maximum valid hidden length before window generation, so padded tail frames do not create MaxSim windows.
  - `grad_cache_chunk_size`: control `128`, failed Whisper run `256`, this relaunch `128`
  - `eval_steps_sample`: control `80` -> ablation `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

First check is operational: step 1 should pass without the Whisper length error, gc256 OOM, or padded-window concatenation OOM. If it trains, compare early dev/ACL/medicine readouts against the Qwen3-Omni/BGE-M3 control and watch step time, since Whisper's forced 3000-frame encoder input may be slower than Qwen3-Omni for 5.76s audio.

## Verdict

FAILED on Slurm 45254 / W&B `prft06bl` before completing step 1. The run passed W&B init, dataset setup, and the initial hard-negative refresh started, but OOMed in `_maxsim_score_mfa` at `sim_all = torch.matmul(speech_embs, text_embs.T)` while comparing the per-rank speech windows against the 1024 per-sample hard negatives. The attempted allocation was 26.25 GiB per rank with only about 25.9 GiB free, so the next relaunch should reduce the per-rank batch or hard-negative fanout rather than only changing GradCache chunk size.
