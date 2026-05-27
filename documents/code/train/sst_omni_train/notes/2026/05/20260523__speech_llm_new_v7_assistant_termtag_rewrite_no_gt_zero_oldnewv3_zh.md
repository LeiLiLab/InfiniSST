# New V7 assistant term-tag rewrite data, zh

## Hypothesis

New V6 exact-only assistant tagging misses important streaming-boundary terms.  New V7 keeps user input and `term_map` unchanged, but when a GT target translation is not an exact future assistant substring, it may replace a high-overlap local assistant span with the full target translation and wrap it as `<term>...</term>`.

## Background / Motivation

In low-latency inference, retriever lookback can place a long term in the current chunk `term_map` even when the reference translation is split across adjacent assistant chunks.  Exact-only SFT then never teaches the Speech LLM to use those boundary term maps.

## What changed vs baseline

Base data is `new_v5_no_gt_zero_llm_variant_aug_oldnewv3`.

- user input / `term_map`: unchanged
- no-GT chunks: unchanged from New V5
- exact target match: same as New V6
- local rewrite fallback: enabled for target translations with normalized length at least 4
- rewrite rule: replace a high-overlap local assistant span with the full target translation, then wrap the full target as `<term>{translation}</term>`

Examples:

- `·罗斯福` -> `<term>富兰克林·罗斯福</term>`
- `的状态` -> `<term>这种未知的状态</term>`
- `完整视觉体验` -> `<term>完整的视觉体验</term>`

## Expected metrics

Higher assistant term-tag coverage than New V6, especially for streaming-boundary terms.  Downstream eval should strip `<term>` and `</term>` before metric scoring.

## Verdict

Pending.
