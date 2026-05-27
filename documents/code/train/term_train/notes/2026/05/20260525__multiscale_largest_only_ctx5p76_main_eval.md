# Multi-Scale Inference Ablation: Main Retriever On Largest Window Only

## Hypothesis

If the main RASST retriever is forced to use only the largest 5.76s inference
window, recall should drop relative to the variable-context multi-window
readout because the model was trained with MFA-localized term windows.

## Background / Motivation

The paper multi-scale inference ablation compares the main multi-window
retriever against two variants: inference-only largest-window querying, and a
model trained directly on the largest window.  The largest-window trained
variant is already represented by W&B run `d988vg46`, sourced from fixed-5.76s
training run `jyb2u787`.

## What changed vs baseline

This run reuses the main checkpoint from `lh1b88kw` but evaluates it on the
fixed 5.76s dev JSONL:
`/mnt/gemini/home/jiaxuanluo/term_dev_dataset_ctx5p76_new_version.jsonl`.
The metrics denominator is the per-context raw 5.76s glossary:
`documents/code/train/term_train/reports/figures/20260525_dev_raw_glossary_ctx5p76.json`.

## Expected metrics

The direct comparison target is `d988vg46` under the same `ctx5p76` dev-only
protocol.  The multi-window reference remains `q2fus6f1`, which evaluates the
same main checkpoint on the variable-context dev JSONL.

## Verdict

Completed as W&B `g3iayem1`, but this is an auxiliary context-duration
diagnostic rather than the primary MaxSim-window ablation.  It keeps the full
multi-scale MaxSim window set and evaluates the main `lh1b88kw` checkpoint on
the fixed-5.76s dev JSONL.  The primary inference-window ablation is
`y454004y`, where `MAXSIM_WINDOWS=24`.
