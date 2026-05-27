## Hypothesis

Run a temporary JA medicine lm=1 serial job with the final two samples first, then merge those two `instances.log` rows with the first three rows from the original serial job to obtain a faster full-five readout.

## Background / Motivation

The original JA medicine lm=1 serial promptfix job is slow and already paid the vLLM startup cost. Aries GPUs 0,1 are idle, so a temporary tail-first job can overlap the last two samples with the original job's third sample.

## What changed vs baseline

No model, retriever, glossary, or generation setting changes. The temporary input order is changed from `404,545006,596001,605000,606` to `605000,606,404,545006,596001`; final scoring merges rows back into the original order and reruns offline BLEU/StreamLAAL/TERM scoring.

## Expected metrics

Metrics should match a clean five-sample serial run up to stochastic generation differences for samples 605000 and 606. The merged result is temporary rescue evidence until the uninterrupted serial result is complete.

## Verdict

Superseded. The original five-sample JA medicine lm=1 serial run completed and
produced `eval_results.tsv` before the tail-first temporary run produced enough
rows to merge. The tail-first run and merge watcher were stopped to release
Aries GPUs 0,1.
