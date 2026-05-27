## Hypothesis

The hard-medicine glossary currently stores only the manual hard-term evidence
sentence ids. Offline oracle translation needs full sentence-level term maps, so
the glossary should carry every ESO test sentence where each hard term appears.

## Background / Motivation

The normalized hard glossary has 212 unique hard terms, but the stored
`sentence_ids` sum to only 280 evidence occurrences. Current TERM_ACC scoring
uses a fixed raw glossary over all matching source/reference occurrences, so
offline oracle term-map construction can miss many eligible sentences if it
uses these evidence ids directly.

## What changed vs baseline

Backfill the existing glossary in place from
`/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_final/test`.
The script preserves the original manual evidence ids as
`manual_evidence_sample_ids` and `manual_evidence_sentence_ids`, then rewrites
the compatibility `sample_ids` and `sentence_ids` fields to full source-side
occurrences. It also adds paired `sample_sentence_ids`, detailed `occurrences`,
and per-language scorer-aligned ids under `lang_sample_sentence_ids`.
The language-specific ids follow the TERM_ACC scorer rule: for duplicate target
translations, keep only the first source term in glossary order.

## Expected metrics

No model metric is produced by this data-prep step. Expected data diagnostics:
212 entries, 1437 ESO sentences scanned, top-level source occurrence count
larger than the original 280 evidence ids, and scorer-style deduplicated
TERM_TOTAL matching the final-reference denominator for zh/de/ja.

## Verdict

Success. The script scanned 1437 sentences from the final ESO test root and
updated 212 glossary entries in place. The original evidence-only JSON and
stats file were backed up with suffix
`.before_sentence_backfill_20260525T055610`.

Key diagnostics:

- original stored `sentence_ids` sum: 280
- full paired source occurrences: 869
- top-level unique `sentence_ids` sum: 850, because `sentence_id` is only unique
  within each sample; use `sample_sentence_ids` for paired ids
- language-eligible occurrence sums before target-translation dedup:
  zh 753, de 718, ja 746
- scorer-style target-translation-dedup denominators:
  zh 744, de 718, ja 736
- final `lang_sample_sentence_ids` / `lang_occurrence_count` are scorer-aligned:
  zh 744, de 718, ja 736
- target-dedup removed occurrences:
  zh 9, de 0, ja 10
