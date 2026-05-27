## Hypothesis

Using a 10k glossary for the retriever term-map SFT data should reduce term-map
density/noise relative to the 100k V13 data while preserving the same
inference-aligned timeline policy.

## Background / Motivation

V13 with the 100k glossary produced an average of about 9 terms per chunk, which
means `tau=0.73` barely filtered the top-10 candidates.  V14 keeps the same
retriever and timeline policy but uses the 10k zh glossary to test whether a
smaller bank yields cleaner term maps for Speech LLM SFT.

## What changed vs baseline

- Rebuild `gt_terms_by_chunk` from source chunk ASR using the same 10k zh
  glossary used for retrieval.
- Keep only `merge_multiplier=1..6`.
- Use `lookback_sec=1.92`, `min_context_sec=2.88`, and `max_context_sec=5.76`.
- For chunks already at least `2.88s`, retrieve current chunk only.
- For shorter chunks, wait until timeline context reaches `2.88s`.
- Use `tau=0.73`, `top_k=10`, no GT backfill.

## Expected metrics

Expected term-map density should be lower than V13.  GT recall should be
interpreted against the 10k glossary-derived GT terms, not the 100k GT terms.

## Verdict

Pending build.
