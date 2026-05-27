# V7 Refmatch R95 Retriever Term-map Data, zh

## Hypothesis

A 95%-ish GT-term-in-term-map rate better matches the deployed retriever than
the near-oracle V6 high-GT curriculum.  It should keep strong exact-adoption
supervision without overfitting Speech LLM to perfect term maps.

## Background / Motivation

V6 reached 99.88% GT-term-in-term-map rate, which is higher than the expected
real retriever recall.  The user requested a more realistic 95% target.

## What changed vs baseline

Input remains source-match 100k + real retriever timeline data from
`speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519`.

V7 uses `--variant refmatch_r95 --gt-target-match-policy full_ref`:

- only reference-compatible source-match terms are trusted as GT;
- most GT chunks include all filtered GT terms;
- a controlled subset of multi-term chunks drops one GT term to simulate
  imperfect retriever recall;
- no-GT chunks remain mostly empty/sparse.

## Expected metrics

Generated train/dev data should have `gt_term_in_term_map_rate` around 95%,
with non-GT density still much lower than V3.

## Verdict

Success.  The generated data hit the requested 95%-ish GT coverage target.

Generated outputs:

- train: 12,500 rows / 68,705 chunks
- dev: 355 rows / 891 chunks
- train ref-match GT terms kept: 82,068 / 172,726 (47.51%)
- train ref-match GT chunks kept: 39,804 / 53,770 original GT chunks (74.03%)
- train GT-term-in-term-map rate: 94.83%
- dev GT-term-in-term-map rate: 95.48%
- train GT chunk any-term coverage: 100.00%
- train GT chunk all-term coverage: 89.35%
- train no-GT nonempty term-map rate: 17.48%
- train average term-map entries per chunk: 1.64
- train average non-GT entries per chunk: 0.51

This version is the recommended SFT input over V6 because it keeps the
reference-compatible target filter while matching the real retriever's expected
95%-ish recall regime.
