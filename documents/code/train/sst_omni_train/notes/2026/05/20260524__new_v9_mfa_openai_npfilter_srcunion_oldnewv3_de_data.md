## Hypothesis

Using a source glossary that unions wiki100k with de old-new_v3 noun/entity candidates should recover GigaSpeech-source coverage while keeping common-word GT pollution lower than the previous de New V9 build.

## Background / Motivation

The earlier de rebuild mistakenly used pure `p31_untrained` wiki100k for source exact matching. Unlike the zh source glossary, that file does not include GigaSpeech train-side terms, so source-exact GT was artificially sparse.

## What changed vs baseline

- Extract de old-new_v3 noun/entity candidates from the SFT source TSV.
- Build a source-only merged glossary: wiki100k plus filtered de candidates.
- Use the merged source glossary for MFA exact source matching.
- Use the candidate list as a phrase/type allowlist.
- Keep OpenAI exact-span rewrite, old-new_v3 TCM term_map, no-GT-zero, and assistant `<term>` tags.

## Expected metrics

GT term count should be materially higher than pure wiki100k source matching while avoiding the previous one-word/common-term dominance.

## Verdict

Pending data-prep completion and validation.
