# zh Cap16 Denoise-Budget Short-Tag Data

## Hypothesis
Chinese medicine underperforms German/Japanese partly because the current zh SLM data is denser and older than the de/ja cap16 denoise-budget short-tag data. Matching the de/ja data recipe should reduce noisy runtime term maps and align supervision with the current tagged-output evaluator.

## Background / Motivation
The current zh medicine main row uses the new_v9 SLM branch trained before the de/ja cap16 denoise-budget repair. German and Japanese were later moved to HN1024 tau `0.78`, cap16 exact-boundary term maps, 6/8/10 denoise-budget sampling, no-GT emptying, and short `<t>...</t>` assistant tags.

## What changed vs baseline
- Add zh source/glossary support to `20260525__build_deja_termmap_ablation_cap16_exactboundary.sh`.
- Build zh HN1024 tau `0.78` cap16 exact-boundary retriever data from `/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`.
- Rebuild user-side term maps with denoise-budget policy: choices `6,8,10`, weights `0.45,0.35,0.20`, no-GT max terms `4`, no-GT empty probability `0.35`.
- Preserve GT terms, drop noisy non-GT terms by score/support, and wrap assistant target translations as `<t>{translation}</t>`.
- Store large generated outputs under `/mnt/data1/jiaxuanluo` because `/mnt/gemini/data1` is effectively full.

## Expected metrics
This data should be used for a new zh SLM training job matching the de/ja cap16-denoise tagged-term setup. The first readout should be tagged ACL raw zh and medicine hard/raw zh with HN1024 tau `0.78`, omit-empty term maps, and `term_t` tag stripping.

## Verdict
Success. The detached job waited for Taurus GPUs `0,1,2,3`, completed the HN1024 tau `0.78` cap16 exact-boundary build, applied the denoise-budget policy, wrapped assistant targets with `<t>...</t>`, and passed validation.

Final validation:

- Train: `12500` rows, `68705` chunks, `56411` term-map chunks, rate `0.8210610581471509`, max entries `14`, tagged rows `12164`.
- Dev: `355` rows, `1790` chunks, `1456` term-map chunks, rate `0.8134078212290503`, max entries `13`, tagged rows `348`.
- Malformed `<t>` messages: `0`; legacy `<term>` messages: `0`; Latin boundary cuts: `0`.
