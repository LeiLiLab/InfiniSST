## Hypothesis
Staging the JA cap16-denoise HF export on Aries local NVMe reduces vLLM load time for follow-up medicine evals.

## Background / Motivation
The JA cap16-denoise SLM finished training on Taurus, but Aries eval should load from local disk rather than repeatedly reading the HF shard set from shared storage.

## What changed vs baseline
This maintenance job waits for the completed HF export and copies it to `/mnt/data3/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf`.

## Expected metrics
No model metrics. Success means `config.json`, `generation_config.json`, `model.safetensors.index.json`, 15 safetensor shards, and `.stage_complete` exist in the Aries local cache.

## Verdict
Pending.
