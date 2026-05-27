# ja Cap16 Denoise-Budget Short-Tag Data

## Hypothesis
Japanese cap16 retriever SLM data should benefit from the same denoise-budget and short-tag supervision that replaced the German cap16 SLM branch.

## Background / Motivation
The existing Japanese cap16 branch uses HN1024 retriever term maps capped at 16 entries. The German repair reduced runtime term-map density through a 6/8/10 mixed budget, score-aware dropout for non-GT retrieved terms, no-GT emptying, and short `<t>...</t>` assistant tags.

## What changed vs baseline
- Parent data event: `20260525T0348__data_prepare__deja_termmap_ablation_cap16_exactboundary`.
- Input branch: `/mnt/gemini/data1/jiaxuanluo/speech_llm_deja_termmap_ablation_cap16_exactboundary_20260525/ja/retriever_hn1024_tau078_cap16_exactboundary`.
- Rebuild user-side term maps from `retriever_results_by_chunk` with denoise-budget policy: choices `6,8,10`, weights `0.45,0.35,0.20`, no-GT max terms `4`, no-GT empty probability `0.35`.
- Preserve GT terms, drop noisy non-GT terms according to retriever score and future-assistant support, and wrap assistant target translations as `<t>{translation}</t>`.
- Eval must strip `term_t` output tags before BLEU and latency scoring.

## Expected metrics
This data is intended for a Taurus4 r32/a32 SLM training job. The first gate after export should be tagged ACL raw Japanese with HN1024, tau `0.78`, omit-empty term maps, and short-tag stripping.

## Verdict
Success. Data generation completed after correcting the validation threshold from de-specific `<=12` to cap16-compatible `<=16`; no generated rows were accepted until validation passed.

Final validation:

- Train: `12500` rows, `53352` chunks, `40876` term-map chunks, rate `0.7661568451042136`, max entries `14`, tagged rows `11091`.
- Dev: `355` rows, `1475` chunks, `1098` term-map chunks, rate `0.744406779661017`, max entries `10`, tagged rows `303`.
- Malformed `<t>` messages: `0`; legacy `<term>` messages: `0`; Latin boundary cuts: `0`.
