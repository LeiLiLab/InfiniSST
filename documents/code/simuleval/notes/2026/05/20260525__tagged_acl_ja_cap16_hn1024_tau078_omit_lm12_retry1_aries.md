# Tagged ACL ja cap16 HN1024 tau0.78 omit lm1/lm2 retry1

## Hypothesis

Retrying `lm=1,2` on currently idle Aries GPU pairs may recover the missing
Japanese tagged ACL cap16 readouts after the first attempt failed before
artifact creation.

## Background / Motivation

The parent `lm=1..4` batch completed only `lm=3,4`. `lm=1,2` exited during
vLLM KV-cache initialization with `ShmRingBuffer` shared-memory errors and
wrote no `eval_results.tsv` or `instances.log`. Taurus is the current shell
host but all local GPUs are busy. Aries GPU pair 6,7 is idle and pair 4,5 is
occupied by a DE no-RAG readout, so this retry uses Aries auto-pair mode over
pairs 4,5 and 6,7.

## What changed vs baseline

- Same JA retriever-cap16 exact-boundary r32/a32 ep4 HF model.
- Same HN1024 checkpoint, tau=0.78, top-k=10, and 1.92s lookback.
- Same tagged ACL raw glossary and `empty_term_map_policy=omit`.
- Scope is only the missing `lm=1,2`; `lm=3,4` remain from the parent event.
- Runtime placement uses auto-pair polling, so `lm=2` may wait for either pair
  6,7 to free after `lm=1` or pair 4,5 to free from the DE readout.

## Expected metrics

Produce verified `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log` for `lm=1` and `lm=2`. This is an ACL held-out
readout only and must not be used for calibration or model selection.

## Verdict

Cancelled before completion on 2026-05-25 after Taurus GPU resources became
available. The Aries controller was terminated to avoid duplicating `lm=1,2`;
the successor event is
`20260525T1338__simuleval__tagged_acl_ja_cap16_hn1024_tau078_omit_lm12_retry2_taurus`.
