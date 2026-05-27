# Speech LLM V16 LLM-Variant r32/a64 TP2

## Hypothesis

Increasing LoRA capacity from r8/a32 to r32/a64 may improve term-map adoption on the V16 LLM-variant data.

## Background / Motivation

The r8/a32 V16 LLM-variant model is available and has quick tagged ACL readouts.  This run tests whether capacity, rather than data construction alone, limits adoption.

## What changed vs baseline

- Data: unchanged V16 LLM-variant retriever timeline data.
- LoRA: r32/a64.
- Parallelism: TP=2 with sequence parallel to avoid the r32 TP1 OOM pattern.
- Max length: 4096 in the relaunch.  The first 3072-token attempt failed during strict preprocessing because one row encoded to length 3097, so no samples are silently dropped.

## Expected metrics

Downstream check is tagged ACL `zh lm=2` with fixed raw denominator, runtime raw and gs10k, plus paper110 extracted and extracted-gs10k controls.

## Verdict

Pending training and eval.
