# V6 Refmatch High-GT Retriever Term-map Data, zh

## Hypothesis

After target-match filtering, the remaining GT terms are reliable enough to
train with a 90%+ GT-term-in-term-map rate.  This should better match the real
retriever's high strict-term recall while avoiding reference-conflicting target
translations.

## Background / Motivation

V5 refmatch fixed the target-translation conflict, but its
`gt_term_in_term_map_rate` was only about 78%.  The deployed retriever typically
recalls strict terms at 90%+, so the SFT data should expose the Speech LLM to a
similar high-GT-recall regime.

## What changed vs baseline

Input remains the source-match 100k + real retriever timeline data from
`speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519`.

V6 uses `--variant refmatch_higt --gt-target-match-policy full_ref`:

- only reference-compatible source-match terms are trusted as GT;
- GT chunks usually include all filtered GT terms, optionally with sparse real
  retriever noise;
- no-GT chunks remain mostly empty/sparse to keep empty term-map robustness.

## Expected metrics

The generated train/dev data should have `gt_term_in_term_map_rate >= 90%`.
If trained, it should improve zh exact TERM_ACC relative to V5/V3 without
requiring Gemini-generated target rewrites.

## Verdict

Success.  The generated data exceeds the 90% target.

Generated outputs:

- train: 12,500 rows / 68,705 chunks
- dev: 355 rows / 891 chunks
- train ref-match GT terms kept: 82,068 / 172,726 (47.51%)
- train ref-match GT chunks kept: 39,804 / 53,770 original GT chunks (74.03%)
- train GT-term-in-term-map rate after filtering: 99.88%
- dev GT-term-in-term-map rate after filtering: 100.00%
- train GT chunk any-term coverage: 100.00%
- train GT chunk all-term coverage: 99.82%
- train no-GT nonempty term-map rate: 17.38%
- train average term-map entries per chunk: 1.62
- train average non-GT entries per chunk: 0.43

This version is the recommended replacement for V5 if the next SFT should
match the real retriever's 90%+ recall regime.
