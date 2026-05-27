# New V9 assistant term-tag delay-clean data, zh

## Hypothesis

New V9 keeps the useful New V7 assistant-side term salience signal while removing dirty source terms and handling split target translations by delaying the earlier prefix into the later chunk.

## Background / Motivation

Some GT terms in the source glossary are not real terminology, especially terms containing pronouns or deictics such as `this`, `that`, `his`, `your`, `what`, and `which`.  Also, local rewrite must handle suffix-only chunks like `·罗斯福` carefully when the prefix was translated at the end of the previous chunk.

## What changed vs baseline

Base data is `new_v5_no_gt_zero_llm_variant_aug_oldnewv3`.

- user input / `term_map`: unchanged
- no-GT chunks: unchanged from New V5
- GT supervision skip: source terms containing pronoun/deictic/WH tokens are removed from assistant-tag supervision
- exact target tag: same as New V6
- local rewrite fallback: same as New V7, with boundary-overlap guard
- boundary delay: if the missing prefix of a target translation is exactly at the previous assistant boundary, remove it there and put the full tagged target in the current assistant

Examples that should now be skipped:

- `this unknowing place => 这种未知的状态`

Examples that may still be rewritten:

- `富兰克林` + `·罗斯福` -> previous chunk drops `富兰克林`, current chunk becomes `<term>富兰克林·罗斯福</term>`
- `完整视觉体验` -> `<term>完整的视觉体验</term>`
- `学生的餐费债务` -> `<term>学生餐费债务</term>`

## Expected metrics

Cleaner assistant term supervision than New V7, with fewer artificial overlaps and less pronoun-term noise.  Downstream eval should strip `<term>` and `</term>` before metric scoring.

## Verdict

Pending.
