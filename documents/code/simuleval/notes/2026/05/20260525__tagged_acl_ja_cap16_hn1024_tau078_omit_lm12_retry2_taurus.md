# Tagged ACL ja cap16 HN1024 tau0.78 omit lm1/lm2 retry2 Taurus

## Hypothesis

Rerunning the missing Japanese `lm=1,2` tagged ACL cap16 readouts on newly
available Taurus GPU pairs should recover the artifacts that failed in the
parent batch.

## Background / Motivation

The parent JA `lm=1..4` batch completed `lm=3,4` only; `lm=1,2` failed during
vLLM shared-memory initialization before any eval artifacts were created. An
Aries retry was started, but it was cancelled before completion after Taurus
GPU pairs 2,3 and 4,5 became available.

## What changed vs baseline

- Same JA retriever-cap16 exact-boundary r32/a32 ep4 HF model.
- Same HN1024 checkpoint, tau=0.78, top-k=10, and 1.92s lookback.
- Same tagged ACL raw glossary and `empty_term_map_policy=omit`.
- Scope is only `lm=1,2`; `lm=3,4` remain from the parent batch.
- Placement: Taurus fixed pairs `lm1=2,3`, `lm2=4,5`.

## Expected metrics

Produce verified `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log` for `lm=1` and `lm=2`. This is an ACL held-out
readout only and must not be used for calibration or model selection.

## Verdict

Failed before eval artifacts on 2026-05-25. Both `lm=1` and `lm=2` crossed the
old shared-memory initialization point, but parallel Taurus vLLM startup failed
during KV-cache setup with `No available memory for the cache blocks`. Taurus
was not actually clean-idle: a co-resident DE cap16 HF export process held GPU
memory across GPUs 1-6. Successor event:
`20260525T1344__simuleval__tagged_acl_ja_cap16_hn1024_tau078_omit_lm12_retry3_taurus_serial_highmem`.
