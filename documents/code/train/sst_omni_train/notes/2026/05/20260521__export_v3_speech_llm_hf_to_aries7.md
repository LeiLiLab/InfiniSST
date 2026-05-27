# Export V3 Speech LLM Checkpoints To Aries7 HF

## Hypothesis

The V3 speech-LLM SFT runs produced valid MCore checkpoints but failed during HF export because `/mnt/gemini/data2` was full. Re-exporting the MCore checkpoints to `/mnt/aries/data7/jiaxuanluo/slm/` should produce eval-ready HF directories without retraining.

## Background / Motivation

Tagged ACL simuleval uses vLLM/HF model paths. The three V3 variants need complete HF exports before running the 3x3 tagged ACL probe requested on 2026-05-21.

## What changed vs baseline

- Baseline run URL: n/a, export recovery for existing SFT runs.
- Diff: no training changes; export-only recovery from MCore checkpoints:
  - `sst_omni/k2xo8quk` real retriever term map, iter 793
  - `sst_omni/891gnubx` tagged term map, iter 923
  - `sst_omni/2st6uspm` adversarial/noisy term map, iter 803
- HF output root: `/mnt/aries/data7/jiaxuanluo/slm/v3_speech_llm/`

## Expected metrics

No model metrics are produced by this event. Success criterion is complete HF export directories with config and safetensor shards.

## Verdict

Success. All three V3 MCore checkpoints were exported to complete HF directories under `/mnt/aries/data7/jiaxuanluo/slm/v3_speech_llm/` with 15 safetensor shards each.
