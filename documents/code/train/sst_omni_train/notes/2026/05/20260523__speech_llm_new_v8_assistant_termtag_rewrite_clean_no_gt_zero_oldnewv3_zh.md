# New V8 assistant term-tag rewrite-clean data, zh

## Hypothesis

New V8 keeps the useful New V7 assistant-side term salience signal while removing dirty source terms and unsafe streaming-boundary rewrites.

## Background / Motivation

Some GT terms in the source glossary are not real terminology, especially terms containing pronouns or deictics such as `this`, `that`, `his`, `your`, `what`, and `which`.  Also, local rewrite must not convert a suffix-only chunk like `·罗斯福` into the full term `富兰克林·罗斯福` when the prefix was already translated in a previous chunk.

## What changed vs baseline

Base data is `new_v5_no_gt_zero_llm_variant_aug_oldnewv3`.

- user input / `term_map`: unchanged
- no-GT chunks: unchanged from New V5
- GT supervision skip: source terms containing pronoun/deictic/WH tokens are removed from assistant-tag supervision
- exact target tag: same as New V6
- local rewrite fallback: same as New V7, but with boundary-overlap guard
- boundary guard: if the missing prefix of a target translation already appears in previous assistant text, do not rewrite the current suffix into the full target

Examples that should now be skipped:

- `this unknowing place => 这种未知的状态`
- `Franklin Roosevelt => 富兰克林·罗斯福` when current assistant only has `·罗斯福` and previous assistant already has `富兰克林`

Examples that may still be rewritten:

- `完整视觉体验` -> `<term>完整的视觉体验</term>`
- `学生的餐费债务` -> `<term>学生餐费债务</term>`

## Expected metrics

Cleaner assistant term supervision than New V7, with fewer artificial overlaps and less pronoun-term noise.  Downstream eval should strip `<term>` and `</term>` before metric scoring.

## Verdict

Pending.
