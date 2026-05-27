## Hypothesis
The JA cap16-denoise run finished training, but the HF export failed because a stale staging directory already existed. Re-running export with a fresh stage root should produce the HF directory needed by Aries vLLM eval.

## Background / Motivation
The training log reports `run_dir=/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_ja_cap16_denoise_budget_ttag_r32a32_ep1_taurus4/keep1.0_r32/v2-20260525-235251` and intended `hf_dir=.../v2-20260525-235251-hf`, but no complete HF export exists yet.

## What changed vs baseline
This maintenance retry uses the existing MCore checkpoint and writes the same HF output path, using a fresh local staging root instead of the stale `.stage.31` path.

## Expected metrics
No model metrics. Success means the HF output directory validates with config files, `model.safetensors.index.json`, and 15 safetensor shards.

## Verdict
Pending.
