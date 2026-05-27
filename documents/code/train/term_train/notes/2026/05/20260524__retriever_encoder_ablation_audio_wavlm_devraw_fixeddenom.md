# Retriever Encoder Ablation: WavLM Audio + BGE-M3 Text

## Hypothesis

Replacing the Qwen3-Omni audio encoder with WavLM should test whether the main
speech encoder is necessary for robust glossary retrieval under large runtime
banks.

## Background / Motivation

The `cyzz2lw0` Babel checkpoint is the WavLM speech-encoder ablation checkpoint
supplied for the paper comparison.  Existing inline metrics are sampled and do
not include a matched full-dev gs100k point.

## What changed vs baseline

This eval keeps BGE-M3 text encoding and replaces the audio encoder with
`microsoft/wavlm-large`.  It preserves the WavLM-specific MaxSim window schedule
and LoRA target modules used by the checkpoint.  The eval is dev-only and uses
the explicit full-dev raw glossary as the fixed metric denominator.

## Expected metrics

The matched full-dev eval should show whether WavLM preserves retrieval recall
as the runtime bank expands from raw to gs10k and gs100k.

## Verdict

PENDING: fill after the matched dev-only eval finishes and the Matplotlib figure
is generated.
