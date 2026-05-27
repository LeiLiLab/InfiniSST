## Hypothesis

Dense top10 no-tau retriever term maps with GT backfill should recover the useful behavior of the older `new_v3` Speech LLM SFT while using the current `lh1b88kw` retriever and exact-reference GT filtering.

## Background / Motivation

V7/V8 refmatch-clean data underperformed the old dense `new_v3` checkpoints on tagged ACL zh lm2/raw.  This run tests whether the main missing ingredient is the training term-map distribution rather than LoRA rank or XML tagging.

## What changed vs baseline

- Train data: V9 dense top10 no-tau retriever term maps.
- Term-map cap: 20 entries per chunk.
- GT terms: exact target substring in assistant reference required before backfill.
- LoRA: rank 8, alpha 32.
- Base model: Qwen3-Omni MCore initial checkpoint.

## Expected metrics

On quick tagged ACL zh lm2/raw, V9 should beat V7/V8 TERM_ACC and ideally approach or exceed old `new_v3` while keeping TERM_FCR below no-TM-SFT.

## Verdict

Pending.
