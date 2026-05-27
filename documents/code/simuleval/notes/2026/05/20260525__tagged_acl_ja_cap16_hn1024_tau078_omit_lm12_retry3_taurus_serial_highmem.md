# Tagged ACL ja cap16 HN1024 tau0.78 omit lm1/lm2 retry3 Taurus serial highmem

## Hypothesis

Running the missing JA `lm=1,2` readouts serially on one clean Taurus 2-GPU pair
should avoid the KV-cache startup failure seen in the parallel Taurus retry.

## Background / Motivation

`lm=1,2` are still missing from the JA cap16 tagged ACL readout. The parent
batch failed before artifacts with a shared-memory error. The Taurus parallel
retry crossed that failure point but failed with no KV-cache memory because a
co-resident DE cap16 HF export process was using GPU memory on Taurus GPUs
1-6.

## What changed vs baseline

- Same JA retriever-cap16 exact-boundary r32/a32 ep4 HF model.
- Same HN1024 checkpoint, tau=0.78, top-k=10, and 1.92s lookback.
- Same tagged ACL raw glossary and `empty_term_map_policy=omit`.
- Scope is only `lm=1,2`; `lm=3,4` remain from the parent batch.
- Placement: Taurus pair `0,7`, serial `lm1` then `lm2`.
- Runtime resource choice: use the normal RAG setting
  `GPU_MEMORY_UTILIZATION_OVERRIDE=0.72`; the pair is clean and avoids the
  co-resident export on GPUs 1-6.

## Expected metrics

Produce verified `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log` for `lm=1` and `lm=2`. This is an ACL held-out
readout only and must not be used for calibration or model selection.

## Verdict

Pending. Metrics and final status will be taken from verified output artifacts
and W&B eval runs, then recorded in the retry event manifest.
