# Analysis: Multi-Scale Inference Ablation Report

## Hypothesis

Summarize the verified dev readouts for the paper multi-scale inference
ablation without introducing hand-written metric truth outside W&B/manifests.

## Background / Motivation

The ablation compares the main multi-scale MaxSim retriever, an inference-only
largest-window MaxSim variant, and a dense 1.92s single-embedding trained
variant.  The dense training source predates event manifests and is backfilled
from W&B/filesystem evidence.

## What changed vs baseline

This analysis writes a TSV and markdown report from already completed W&B runs
and manifests.  It does not launch training or choose hyperparameters.

## Expected metrics

The report should contain run ids, protocols, manifest paths, and the dev
recall keys needed for the ablation table.

## Verdict

Completed.  Outputs are:

```text
documents/code/train/term_train/reports/figures/20260525_multiscale_inference_ablation_devraw.tsv
documents/code/train/term_train/reports/20260525_multiscale_inference_ablation_devraw.md
```
