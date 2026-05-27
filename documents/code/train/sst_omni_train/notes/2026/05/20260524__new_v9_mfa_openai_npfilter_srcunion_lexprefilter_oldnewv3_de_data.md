## Hypothesis

Adding a legacy term-map exact-span prefilter should keep clean source/ref exact evidence while reducing OpenAI rewrite calls from all source candidates to candidates whose existing de translation is actually supported by the assistant text.

## Background / Motivation

Pure wiki100k missed GigaSpeech train-side terms. The first source-union attempt fixed source coverage but produced about 113k OpenAI candidate rewrites, which is too slow and expensive.

## What changed vs baseline

- Build source glossary from wiki100k plus de old-new_v3 noun/entity candidates.
- Require MFA source exact match and old-new_v3 source candidate allowlist.
- Use legacy input term_map only as a de translation lexicon: candidate survives only if a legacy term-map translation appears as an exact future assistant substring.
- OpenAI then rewrites that exact evidence-backed span into an uncommon term translation.
- Continue with old-new_v3 TCM term_map, GT backfill, no-GT-zero, and assistant `<term>` tags.

## Expected metrics

OpenAI pending count should drop by at least an order of magnitude versus the unfiltered source-union attempt, while GT term count should remain higher and cleaner than pure wiki100k matching.

## Verdict

Pending data-prep completion and validation.
