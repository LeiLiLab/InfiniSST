# Export V16 LLM-Variant R32 HF To Aries6

## Hypothesis

The V16 LLM-variant r32/a64 training checkpoint is valid, but the first HF
export failed because `/mnt/aries/data7` was full.  Re-exporting the existing
MCore checkpoint to `/mnt/aries/data6` should produce a complete eval-ready HF
directory without retraining.

## Background / Motivation

The training run `pqg310ag` finished and saved MCore checkpoint iteration 719,
but `swift export --to_hf` failed with `No space left on device` while writing
safetensor shards to data7.  The partial HF directory contains only 7 of 15
expected shards, so simuleval must not use it.

## What changed vs baseline

- No training changes.
- Source MCore checkpoint: the completed r32/a64 V16 LLM-variant checkpoint on
  data7.
- Destination HF export: data6, which has free space.
- Export tool: `swift export --mcore_adapters --to_hf true`.

## Expected metrics

No model metrics are produced.  Success criterion is a complete HF export with
config files and all safetensor shards.

## Verdict

Export succeeded on data6.  The HF directory contains all 15 expected
safetensor shards and is ready for simuleval.
