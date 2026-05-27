## Hypothesis

V7/V8 were too sparse and oracle-like.  A better retriever-SFT distribution should mimic the older successful `new_v3` data: dense top-k term maps, no tau filtering during SFT data construction, cap around 20 entries, and exact-reference GT backfill.

## Background / Motivation

Quick tagged-ACL zh lm2/raw eval showed old `new_v3` checkpoints outperform V7/V8 and sharply reduce false copy rate.  The next data version therefore removes the tau filter from training term-map construction and restores dense retriever exposure.

## What changed vs baseline

- Retriever source: `lh1b88kw` timeline retrieval, `top_k=10`, score threshold disabled.
- GT supervision: source-glossary exact-match terms are trusted only when the target translation is an exact substring of the assistant reference.
- V9: top10 retriever entries plus exact-reference GT backfill, capped at 20 terms per chunk.
- V10: same as V9, but GT target translations are wrapped with deterministic random marker strings in both term_map and assistant targets.

## Expected metrics

For data stats, expect dense nonempty term maps and roughly top10 average retriever entries per chunk.  V10 should report nonzero assistant marker replacements.

## Verdict

Pending.
