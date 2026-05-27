# Retriever Speech Encoder Ablation: Whisper Medium.en on Varctx576

## Hypothesis

Replacing the default Qwen3-Omni audio encoder with `openai/whisper-medium.en`, while keeping the BGE-M3 text encoder and the same varctx576 retriever training data, may change speech-side alignment quality and training speed enough to justify a deeper audio-encoder ablation.

## Background / Motivation

The BGE-large text-encoder ablation underperformed the BGE-M3 control in the paired epoch-0 eval-only readout, so the next useful axis is the speech encoder. The current control run uses Qwen3-Omni audio features with BGE-M3 text embeddings, varctx576 data, MaxSim MFA supervision, hard-neg depth 1024, and TCM disabled.

This run keeps the faster eval policy from the current Taurus ablations: `grad_cache_chunk_size=256`, `eval_steps_sample=100`, dev eval sampled to 100 examples, and ACL/medicine kept as full inline readouts with only `tau=0.75` threshold diagnostics.

## What changed vs baseline

- Baseline run URL: https://wandb.ai/luojiaxuan1215-johns-hopkins-university/qwen3_rag/runs/lh1b88kw
- Baseline anchor: `best_secondary/step=2640`, Qwen3-Omni audio encoder, BGE-M3 text encoder, varctx576, TCM-off.
- Diff:
  - `audio_encoder_preset`: `qwen3-omni` -> `whisper-mid-en`
  - `audio_encoder_type`: `qwen3_omni` -> `whisper`
  - `audio_model_id`: `Atotti/Qwen3-Omni-AudioTransformer` -> `openai/whisper-medium.en`
  - `audio_feature_extractor_id`: `openai/whisper-large-v3` -> `openai/whisper-medium.en`
  - `lora_target_modules`: Qwen targets including `proj1/proj2` -> Whisper targets `q_proj k_proj v_proj out_proj fc1 fc2`
  - `grad_cache_chunk_size`: control `128` -> ablation `256`
  - `eval_steps_sample`: control `80` -> ablation `100`, aligned as a multiple of `NEG_BANK_REFRESH_STEPS=50`
  - `eval_glossary_sizes`: dev includes `1000 10000`; ACL and medicine use `10000`
  - `tcm_sweep_thresholds`: only `0.75`, with raw recall representing tau 0.0

## Expected metrics

Primary comparison is against the Qwen3-Omni/BGE-M3 control at matched early training steps and then at best-secondary checkpoints if the run survives long enough. A useful result would retain dev `recall@10_gs10000` within roughly 1 pp of the control while improving ACL/medicine readouts or reducing step time.

## Verdict

FAILED on Slurm 45250 / W&B `o0p64pr7` before completing step 1. The first launcher exposed a Whisper interface mismatch: `openai/whisper-medium.en` requires 3000-frame mel input, but the 5.76s varctx collate path emitted 576-frame mel features. The follow-up fix pads/truncates Whisper mel features to 3000 inside `qwen3_glossary_neg_train.py` and preserves the true valid length for pooling masks.
