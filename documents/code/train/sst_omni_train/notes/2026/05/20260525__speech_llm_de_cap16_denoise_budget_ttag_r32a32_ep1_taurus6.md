# Speech LLM Training: de Cap16 Denoise-Budget Short-Tag r32/a32 Taurus6

## Hypothesis
The de cap16 denoising-budget short-tag data should improve SLM robustness to noisy runtime term maps while reducing output tag overhead via `<t>...</t>` supervision.

## Background / Motivation
The 8-GPU watcher was waiting on GPU 0.  Taurus currently has six contiguous idle GPUs available, so this run switches to a 6-GPU topology to start immediately rather than waiting for all eight cards.

## What changed vs baseline
- Data is unchanged from `20260525T1225__data_prepare__de_cap16_denoise_budget_ttag`.
- LoRA is unchanged: r32/a32.
- Max length remains 3072.
- Parallelism changes from the planned 8-GPU launcher to the existing Taurus6 pattern: `NPROC=6`, `EP=2`, `TP=2`, `sequence_parallel=true`, `GLOBAL_BATCH_SIZE=6`.
- Runtime eval must use `--strip-output-tags term_t`.

## Expected metrics
The immediate goal is to produce a usable checkpoint quickly.  First eval gate remains tagged ACL raw de with HN1024 and short-tag stripping, checking BLEU recovery while preserving TERM_ACC above no-RAG.

## Verdict
Pending.  Training submitted directly on Taurus GPUs 1,2,3,4,5,6.
