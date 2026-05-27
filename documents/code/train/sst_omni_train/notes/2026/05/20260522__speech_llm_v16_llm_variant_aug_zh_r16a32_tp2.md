# Speech LLM V16 LLM-Variant r16/a32 TP2

## Hypothesis

A middle LoRA capacity, r16/a32, may recover some adoption benefit over r8/a32 without the instability or overfitting risk of larger ranks.

## Background / Motivation

V16 LLM-variant r8/a32 is the current low-capacity reference.  The user requested rank-capacity checks for r32/a64 and r16/a32.

## What changed vs baseline

- Data: unchanged V16 LLM-variant retriever timeline data.
- LoRA: r16/a32.
- Parallelism: TP=2 with sequence parallel, matched to the r32/a64 capacity run.
- Max length: 4096 in the relaunch.  The first 3072-token attempt failed during strict preprocessing because one row encoded to length 3097, so no samples are silently dropped.

## Expected metrics

Downstream check is tagged ACL `zh lm=2` with fixed raw denominator, runtime raw and gs10k, plus paper110 extracted and extracted-gs10k controls.

## Verdict

Pending training and eval.
