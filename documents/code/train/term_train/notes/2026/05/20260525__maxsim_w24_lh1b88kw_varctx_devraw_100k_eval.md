# Multi-Scale Inference Ablation: MaxSim Largest Window Only

## Hypothesis

If the main RASST retriever is forced to score only the largest internal
MaxSim speech window (`24` frames), recall should drop relative to the
multi-scale MaxSim readout because the trained model and inference recipe no
longer aggregate evidence across the smaller localized windows.

## Background / Motivation

Section 4.2 describes RASST inference as querying speech windows at multiple
scales and keeping the highest-similarity term matches.  The main verified
dev-only reference for this protocol is W&B run `q2fus6f1`, using checkpoint
source run `lh1b88kw` and `MAXSIM_WINDOWS="2 3 4 5 6 7 8 10 12 16 20 24"`.

## What changed vs baseline

This readout keeps the same `lh1b88kw` checkpoint, variable-context dev JSONL,
retrieval bank, fixed raw denominator, scoring chunks, and dev-only scope as
`q2fus6f1`.  The only intended inference change is:

```text
MAXSIM_WINDOWS=24
```

The launcher was patched to allow `MAXSIM_WINDOWS` to be overridden while
retaining the multi-scale default for existing context-ablation runs.

## Expected metrics

Primary comparison is against `q2fus6f1` on the same varctx dev protocol.
Metrics should be read from W&B/history and mirrored into the manifest after
completion.  ACL, tagged ACL, and medicine readouts are disabled; this is not a
model-selection run.

## Verdict

Completed as W&B `y454004y`.  Under the varctx dev fixed-raw protocol,
`MAXSIM_WINDOWS=24` reduced recall relative to the multi-scale reference
`q2fus6f1`: `eval_dev/recall@10_gs100000` was `0.9607` vs `0.9858`, and
base-bank `eval_dev/recall@10` was `0.9821` vs `0.9920`.
