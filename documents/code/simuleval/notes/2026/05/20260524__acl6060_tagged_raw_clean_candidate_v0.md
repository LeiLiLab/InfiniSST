# ACL6060 Tagged Raw Clean Candidate V0

## Hypothesis

The released tagged ACL raw glossary contains generic one-word academic terms and identity-copy acronyms that are weak terminology targets.  A conservative clean candidate can help diagnose whether TERM_ACC trends are driven by strict terminology or noisy denominator terms.

## Background / Motivation

This is an analysis candidate only.  It must not silently replace the fixed raw glossary denominator used in main results.

## What changed vs baseline

Starting from `acl6060_tagged_gt_raw_min_norm2.json`, remove entries with at least one of:

- `identity_copy_zh`: Chinese target exactly equals the English term, case-insensitive.
- `generic_one_word`: one-word common academic term such as model, task, data, language, question, method.
- `too_short_zh`: Chinese target length is <=1, excluding common acronym allowlist.

## Expected metrics

No metrics are produced by this event.  The output is a candidate glossary plus removed-term TSV for manual review.

## Verdict

Created candidate glossary with 175 kept terms and 63 removed terms.  Use only as a diagnostic/appendix candidate unless the paper protocol explicitly changes.
