# Export New V5 no-GT-zero old-new_v3 R32 HF To Aries6

## Hypothesis

The New V5 training process failed after checkpoint save due data7 disk pressure,
but the iteration-1000 MCore checkpoint is usable.  Re-exporting that checkpoint
to data6 should create a complete HF model directory for simuleval.

## Background / Motivation

W&B run `cg5qisu9` failed with `No space left on device` from TensorBoard event
writing.  The MCore checkpoint directory contains `iter_0001000` and
`latest_checkpointed_iteration.txt=1000`, so evaluation can proceed after a
clean HF export.

## What changed vs baseline

- No training changes.
- Source MCore checkpoint: New V5 no-GT-zero old-new_v3 r32/a64 TP2 checkpoint.
- Destination HF export: data6, because data7 is full.
- Export tool: `swift export --mcore_adapters --to_hf true`.

## Expected metrics

No model metrics are produced.  Success criterion is a complete HF export with
config files and all 15 safetensor shards.

## Verdict

Export succeeded on data6.  The HF directory contains all 15 expected
safetensor shards and is ready for simuleval.
