# Retriever Encoder Ablation: Qwen3-Omni Audio + Multilingual-E5 Text

## Hypothesis

Replacing BGE-M3 with multilingual-E5 should test whether the main text encoder
is necessary for robust glossary retrieval under large runtime banks.

## Background / Motivation

The `xw53jzn0` Babel checkpoint is the text-encoder ablation checkpoint supplied
for the paper comparison.  Existing inline metrics are not sufficient for the
paper figure because they used sampled dev evaluation and did not include a
matched gs100k point.

## What changed vs baseline

This eval keeps the Qwen3-Omni audio encoder and replaces the text encoder with
`intfloat/multilingual-e5-large`, using `query: ` prefixing and mean pooling.
The eval is dev-only and uses the explicit full-dev raw glossary as the fixed
metric denominator.

## Expected metrics

The matched full-dev eval should show whether the early best E5 checkpoint is
competitive with the BGE-M3 main retriever across raw, gs10k, and gs100k banks.

## Verdict

PENDING: fill after the matched dev-only eval finishes and the Matplotlib figure
is generated.
