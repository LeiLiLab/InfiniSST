## Canceled / Do Not Use

This branch is canceled. It incorrectly followed the v13/sourceexact route.
The final zh Speech LLM lineage is old new_v3 r32 -> new_v4 -> new_v5 ->
new_v9, so de/ja must follow that lineage instead. Generated outputs were
removed and the event manifest is marked `canceled`.

## Hypothesis

de/ja Speech LLM term-map SFT should use the same clean GT policy as zh v13:
source chunk ASR exact glossary matches plus exact future target evidence.

## Background / Motivation

The earlier de/ja New V9 data derived `gt_terms_by_chunk` from existing
LLM-generated `term_map` entries with fuzzy matching.  That polluted GT terms
and made later LLM target variants and assistant `<term>` tags amplify bad
supervision.

## What changed vs baseline

- Do not call `derive_gt_terms_from_termmap_matches.py`.
- Add `source_chunk_asr_by_chunk` from the language TSV `src_trajectory`.
- Build GT terms only when the source term exactly matches the source chunk and
  the target translation exactly appears in current/future assistant text.
- Rebuild retriever timeline term maps from clean GT data.
- Apply LLM target variants, no-GT-zero, and assistant `<term>` tags only after
  clean GT construction.
- Assistant tag repair is exact-only plus adjacent assistant boundary repair;
  global fuzzy rewrite is disabled.

## Expected metrics

- Zero missing TSV rows and zero dropped rows.
- Zero global fuzzy assistant rewrites.
- Zero unbalanced tags and zero Latin word-boundary tag violations.
- Report source exact matches, future-ref GT terms, source==target GT terms,
  term-map GT recall, no-GT term-map density, and assistant tag rates.

## Verdict

Pending.
