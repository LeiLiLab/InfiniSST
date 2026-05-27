# V8 refmatch R95 XML-term data

## Hypothesis

Using the same V7 refmatch R95 curriculum but wrapping each term entry with
`<term>...</term>` will make term-map evidence more salient without changing
the underlying GT/noise distribution.

## Background / Motivation

V7 fixes two earlier data problems: trusted GT terms must have exact target
substring support in the reference, and GT-term-in-term-map recall should be
close to the deployed retriever rather than oracle-like.  The remaining question
is whether explicit XML-style tags help the Speech LLM notice and adopt the
provided term translations.

## What changed vs baseline

- Baseline data event: `20260521T1350__data_prepare__v7_refmatch_r95_termmap_zh`
- Same source JSONL as V7.
- Same `refmatch_r95` sampling variant and seed as V7.
- Same `gt_target_match_policy=full_ref`.
- Only term-map rendering changes from plain `source=target` to
  `<term>source => target</term>`.

## Expected metrics

The generated train/dev stats should match V7's GT/noise distribution, with
`gt_term_in_term_map_rate` in the 94-96.5% range.  Downstream evaluation should
test whether XML tags improve exact TERM_ACC over V7 on tagged ACL `zh lm2/raw`.

## Verdict

Pending.
