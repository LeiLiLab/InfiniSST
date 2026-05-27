## Hypothesis

Adding explicit ACL sentence ids to the tagged ACL6060 glossary will let offline
SLM readouts use the tagged glossary with sentence-aligned oracle term maps
without guessing which sentence introduced each tagged term.

## Background / Motivation

The existing tagged ACL6060 glossary files contain terms and translations, but
not the original ACL sentence id. Offline SLM analysis can consume an explicit
sentence term-map, so the tagged glossary needs a deterministic bridge from
term entries back to ACL6060 source sentences.

## What changed vs baseline

This data-prep step reads the ACL6060 XML segment ids, the tagged English
transcript, and the existing tagged glossary. It writes derived glossary files
with `sentence_ids`, `sentence_indices`, `utter_ids`, and occurrence metadata.
It also writes `sentence_term_map` JSON files for `zh`, `ja`, and `de`.

No original glossary file is overwritten.

## Expected metrics

No model metric is produced by this event. Expected data checks:

- XML, tagged transcript, and plain source transcript must all contain 468
  aligned sentences.
- Tagged source terms with `min_norm_chars=2` should recover the existing 238
  tagged ACL GT terms.
- Language-specific sentence term maps should be directly consumable by
  `documents/code/offline_sst_eval/compute_sentence_term_adoption.py`.

## Verdict

Success. The generated files passed the alignment checks:

- XML, tagged transcript, and source transcript are aligned at 468 sentences.
- The existing raw tagged ACL glossary has 238 entries, and all 238 now have
  sentence ids.
- The 10k tagged+wiki glossary keeps 10,000 entries; the 238 tagged-GT entries
  have sentence ids and the 9,762 wiki filler entries intentionally have empty
  occurrence metadata.
- The source dictionary copy has 241 entries, and all 241 now have sentence
  ids. Three phrase-level entries were explicitly backfilled with exact source
  text matches because the tagged transcript splits them into adjacent bracketed
  components.
- Kept tagged bracket occurrences: 1,163; exact source-text backfilled
  occurrences: 9; total occurrence rows: 1,172 across 409 source sentences.
- The explicit sentence term maps load with
  `compute_sentence_term_adoption.py` for `zh`, `ja`, and `de`.

Primary outputs are under `/mnt/gemini/home/jiaxuanluo/eval_glossaries/`.
