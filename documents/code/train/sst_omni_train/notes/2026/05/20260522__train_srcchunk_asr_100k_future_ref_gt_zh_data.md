## Hypothesis

Exposing the source chunk ASR text and deriving `gt_terms_by_chunk` directly
from that text plus the 100k retriever glossary should give a cleaner and more
inspectable GT source for later Speech LLM term-map construction.

## Background / Motivation

The Speech LLM training JSONL already has streaming chunks.  Before adding any
retriever term map, we need each chunk's source text visible in the data and a
GT term list produced from that source chunk text.

## What changed vs baseline

- Add `source_chunk_asr_by_chunk`, aligned 1:1 with `audios` and
  `gt_terms_by_chunk`.
- Generate `gt_terms_by_chunk` from `source_chunk_asr_by_chunk` exact whole-token
  matches against the 100k retriever glossary.
- Keep only terms whose zh target translation appears as an exact substring in
  assistant messages from the current audio response through the end of the
  conversation.
- Rewrite all existing audio user term maps to `term_map:NONE`; this dataset is
  GT-only and does not inject retriever candidates.
- Train split only.  Dev is intentionally ignored for this step.

## Expected metrics

- `source_chunk_asr_by_chunk` length equals `audios` for every row.
- `gt_terms_by_chunk` length equals `audios` for every row.
- All audio user chunks contain `term_map:NONE`.
- Zero future-reference validation violations.

## Verdict

SUCCESS. Train-only build completed.

- Rows: 12,500.
- Chunks: 68,705.
- Source exact glossary matches before target filtering: 172,726.
- Future-ref GT terms kept: 80,062.
- Chunks with GT: 56.73%.
- Average GT terms per chunk: 1.17.
- Validation passed: `source_chunk_asr_by_chunk`, `gt_terms_by_chunk`, and
  `audios` are aligned; all audio user chunks are `term_map:NONE`; future-ref
  exact violations are zero.
