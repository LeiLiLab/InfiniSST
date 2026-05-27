# Retriever Encoder Ablation: Qwen3-Omni Audio + BGE-M3 Text

## Hypothesis

The main retriever encoder pair should be the strongest reference for the
encoder ablation when evaluated under the same full-dev fixed-denominator
protocol as the ablation checkpoints.

## Background / Motivation

The paper currently has an encoder-ablation placeholder.  This eval creates the
matched main-system point for the Matplotlib figure: full dev JSONL, fixed dev
raw-glossary denominator, and runtime candidate banks raw, gs10k, and gs100k.

## What changed vs baseline

This is an eval-only readout of the `lh1b88kw` main retriever checkpoint.  It
does not train or tune.  ACL, tagged ACL, and medicine are disabled so the plot
is dev-only.

## Expected metrics

Dev Recall@10 should remain close to the previously reported HN1024 main
retriever values, with the denominator fixed to the explicit dev raw glossary
for all bank sizes.

## Verdict

PENDING: fill after the matched dev-only eval finishes and the Matplotlib figure
is generated.
