## Hypothesis

The En-De BLEU shortfall is not explained by aggregate retriever recall alone. We need to inspect the MFA-timed term-map shape under streaming latency multipliers to identify whether BLEU is hurt by prompt exposure mismatch, marginal tau=0.78 terms, stale lookback terms, or SLM sensitivity to dense/noisy maps.

## Background / Motivation

Offline ST improves substantially when given oracle GT terms, but RASST can lose BLEU even with high TERM_ACC. The reviewer-facing question is why runtime term maps help terminology but do not preserve BLEU. This analysis aligns runtime term maps, MFA timestamps, ACL sentence intervals, and eval outputs for De.

## What changed vs baseline

- New analysis script will summarize de/lm=1..4 post-tau HN1024 term-map distributions from existing runtime logs.
- It will separate NewV9/no-GT-zero runs from TM-SFT+HN1024 omit runs where available.
- It will compute per-call and sentence-aligned term-map counts, score bins, stale/current-window ratios, sentence-level gold/noise estimates, and metric correlations.
- No model selection or tau tuning is performed from ACL here; this is failure analysis for SLM/data repair.

## Expected metrics

Expected outputs are TSV/Markdown reports with:

- term-map count and score distributions by lm,
- sentence-level retrieved gold/noise exposure by lm,
- prompt-shape differences between `term_map:NONE` and `empty_term_map_policy=omit`,
- candidate data-adjustment recommendation for the SLM.

## Verdict

Completed. The analysis script writes summary, call-level, sentence-level, and Markdown reports under `documents/code/simuleval/reports/`.

Key finding: for matched lm=2 and lm=4, NewV9 and TM-SFT+HN1024 see identical post-tau term references, so the BLEU difference is SLM/prompt-response behavior rather than retriever output. `empty_term_map_policy=omit` removes `term_map:NONE`, but lm=4 still trails verified no-RAG BLEU (32.5332 vs 33.3008), so NONE blocks are not the only issue.

MFA alignment shows no stale lookback leakage in these runtime logs (`stale_lookback_ref_rate=0`). The main risky shape is local over-exposure: at lm=4, sentence-aligned references have gold recall 0.9269 but sentence-level noise reference rate 0.5632, with 12.67% of references in the low-margin 0.78-0.80 score band. This supports a data-side SLM repair: train with omitted empty maps, keep retrieved maps on no-GT chunks, but add negative/noise exposure and dropout/down-weight low-margin or sentence-unsupported terms instead of forcing adoption.
