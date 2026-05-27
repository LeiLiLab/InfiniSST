# V5 Refmatch Precision Retriever Term-map Data, zh

## Hypothesis

Retriever-SFT should use source-match terms only when the glossary target
translation is compatible with the SFT reference.  Filtering GT/backfill terms
by target exact match should reduce zh exact-translation regressions while
preserving realistic retriever noise exposure.

## Background / Motivation

V3 real/tagged/adv improved robustness in some noisy settings, but zh lm2/raw
lost to no-TM-SFT.  Case analysis showed many misses were exact wording changes
such as `token -> 令牌` becoming `标记/词元`.  V4 precision improved GT density,
but a target-match audit found that only about 64% of GT entries placed in the
term map had target translations that exact-matched the full reference.

## What changed vs baseline

Input remains the source-match 100k + real retriever timeline data from
`speech_llm_retriever_timeline_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260519`.

V5 adds `--gt-target-match-policy full_ref`:

- source terms can still come from the 100k glossary;
- a source-match term is trusted as GT/backfill only if its target translation
  exact-matches the row-level assistant/reference text;
- unmatched source-match terms are demoted to ordinary retriever/noise terms;
- the curriculum otherwise follows the V4 precision distribution.

## Expected metrics

The generated data should keep enough GT signal for SFT while avoiding
reference-conflicting term-map targets.  It should improve zh TERM_ACC relative
to V3/V4-style retriever-SFT and be a cleaner base for later tagged/adv variants.

## Verdict

Success.  Scheme 1 has enough signal and does not require Gemini rewriting for
this first V5 run.

Generated outputs:

- train: 12,500 rows / 68,705 chunks
- dev: 355 rows / 891 chunks
- train ref-match GT terms kept: 82,068 / 172,726 (47.51%)
- train ref-match GT chunks kept: 39,804 / 53,770 original GT chunks (74.03%)
- train GT-term-in-term-map rate after filtering: 77.90%
- train no-GT nonempty term-map rate: 17.67%
- train average term-map entries per chunk: 1.71
- train average non-GT entries per chunk: 0.78

Consistency check passed: every filtered GT term target and every term-map GT
target exact-matches the full row reference in both train and dev.
