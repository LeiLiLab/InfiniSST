# Backfill: Dense 1.92s Single-Embedding Retriever

## Hypothesis

This historical run tested the direct dense single-embedding retriever setting:
the audio side uses transformer pooling over a fixed 1.92s context instead of
multi-scale MaxSim window embeddings.

## Background / Motivation

The run predates the event-manifest workflow.  Provenance here is backfilled
from W&B run `r5l4780c`, local launcher
`documents/code/train/term_train/sweep_text_pooling_aries.sh`, and existing
checkpoint files under `/mnt/aries/data4/jiaxuanluo/train_outputs/sweep_text_pooling`.

## What changed vs baseline

W&B config records `use_maxsim` as absent/disabled, `pooling_type=transformer`,
`text_pooling=cls`, `batch_size=12288`, `temperature=0.03`, and training data
`/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl`.  This is a
historical dense one-shot retriever run, not an MFA-localized MaxSim run.

## Expected metrics

The checkpoint was selected by W&B `best/metric_value=0.8248` for
`eval_acl6060/recall@10_gs1000` at step 198.  The run crashed after the short
sweep budget, so it should be treated as historical-debt evidence rather than a
fully completed training baseline.

## Verdict

Backfilled from W&B/filesystem evidence for use as the dense 1.92s ablation
source.  Missing Slurm job id and exact original log path are left unknown.
