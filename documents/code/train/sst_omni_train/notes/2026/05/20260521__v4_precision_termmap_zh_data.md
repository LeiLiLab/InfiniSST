# V4 Precision Retriever Term-map Data, zh

## Hypothesis

V3 retriever-SFT over-trained robustness/noise and made the zh model ignore or
rewrite exact term-map translations.  A precision-weighted retriever dataset
should preserve no-term robustness while restoring exact term adoption.

## Background / Motivation

Tagged ACL `zh lm2 raw` shows no-TM-SFT beats RASST on TERM_ACC and strongly
beats V3-real.  Error cases are mostly exact-translation preference failures:
`token -> 令牌` becomes `标记/词元`, `transformer -> 转换器` becomes
`Transformer`, and `contextualized -> 情境化` becomes `上下文感知`.

## What changed vs baseline

Baseline data: V3 robust retriever postprocess under
`speech_llm_v3_robust_termmap_zh_lh1b88kw_tau073_srcmatch100k_20260520`.

Diff: reuse the source-match + retriever timeline input, but add a
`precision` curriculum:

- no-GT chunks are mostly empty, with only sparse noise;
- GT chunks are dominated by `term_preference`, `clean_gt`, and small-cap
  realistic retriever maps;
- dense noise is rare;
- `term_preference` prefers GT terms whose target translation appears in the
  local assistant target window.

## Expected metrics

The generated train/dev data should have a higher GT-term-in-term-map rate and
lower non-GT term density than V3-real.  If trained, it should recover zh
TERM_ACC relative to V3-real while preserving the de/ja robustness benefits of
term-map SFT.

## Verdict

Success.  The V4 precision postprocess generated:

- train: 12,500 rows / 68,705 chunks
- dev: 355 rows / 891 chunks
- train GT-term-in-term-map rate: 53.86%
- train GT-chunk any-term coverage: 94.87%
- train no-GT nonempty term-map rate: 17.23%
- train average term-map entries per chunk: 2.18
- train average non-GT entries per chunk: 0.83

Compared with V3-real, this is a more precision-weighted retriever-SFT dataset:
GT term coverage is higher, dense noise is rarer, and no-GT chunks are less
often polluted by nonempty term maps.

Post-hoc target-match check:

- report: `documents/code/train/sst_omni_train/reports/20260521_v4_precision_gt_translation_match_check.md`
- train GT entries placed in term_map whose target exact-matches full row
  reference: 63.65%
- train GT entries placed in term_map whose target exact-matches local target
  window: 60.85%

This means the current V4 data is not recommended as the main training input
without an additional reference-supported GT filter.  The source-match 100k
glossary contains many everyday/dictionary translations that conflict with the
SFT reference wording.
