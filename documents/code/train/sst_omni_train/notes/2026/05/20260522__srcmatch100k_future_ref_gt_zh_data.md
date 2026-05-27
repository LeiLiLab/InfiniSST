## Hypothesis

Rebuilding `gt_terms_by_chunk` from 100k glossary source matches and filtering
by future assistant exact target evidence should produce cleaner Speech LLM
term-map supervision than the historical LLM-extracted GT.

## Background / Motivation

The historical `gt_terms_by_chunk` is noisy, and previous `full_ref` filtering
can keep a term for the current chunk just because the target translation
appeared earlier in the same conversation.  For streaming SFT, the valid
reference evidence for chunk `i` is the assistant response after audio chunk
`i` through the end of the same conversation.

## What changed vs baseline

- Source term authority: 100k glossary exact whole-token match over the
  TSV-derived source chunk text.
- Target-side filter: keep a matched source term only if its zh translation is
  an exact substring in assistant messages from the current audio response to
  the end of the conversation.
- The script validates every emitted GT term after writing the JSONL.

## Expected metrics

- `target_match_policy = future_ref`.
- `future-ref violation = 0`.
- Lower GT density than source-only or full-reference filtering, but cleaner
  positive supervision for REAL_ADOPT training.

## Verdict

DEPRECATED. Full train/dev build completed, but the source-side chunk matching
was based on TSV `src_trajectory` split by the existing streaming chunk count,
not MFA word timestamps.  This output is useful only as a proxy/debug artifact
and must not be used as the final V13 GT source.

- Train: 12,500 rows / 68,705 chunks; 172,726 source-exact glossary matches;
  80,062 future-ref GT terms kept; 56.73% chunks have GT; 1.17 GT terms/chunk.
- Dev: 355 rows / 891 chunks; 1,902 source-exact glossary matches; 813
  future-ref GT terms kept; 50.28% chunks have GT; 0.91 GT terms/chunk.
- Validation: every emitted GT term target translation is an exact substring in
  assistant messages from the current audio response through the end of the
  conversation.
