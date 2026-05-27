## Hypothesis

The cap16 German SLM can recover BLEU if its SFT data teaches the model that retrieved term maps are hints rather than commands. We should keep GT term supervision, but expose the model to smaller and noisier runtime-like term maps where unsupported terms are present and not adopted.

## Background / Motivation

The current cap16 model improves En-De tagged ACL TERM_ACC but low-latency BLEU remains weak. Fixed `max_new_tokens=80` barely changes lm=1/lm=2 BLEU, so the failure mode is not mainly decode truncation. MFA diagnostics indicate that runtime term maps contain many sentence-unsupported terms. This data-prep event creates a denoising/budgeted cap16 variant from the existing HN1024 retriever-result JSONLs.

## What changed vs baseline

- Source data remains the verified cap16 exact-boundary HN1024 tau=0.78 De branch.
- GT terms are always preserved and still define assistant-side `<term>...</term>` targets.
- Non-GT retrieved terms are kept as explicit noise exposure, with score-aware dropout.
- Term-map budgets are mixed across 6/8/10 entries, and no-GT chunks are capped at 4 terms with a 35% empty-map probability.
- A runtime budget schedule is emitted for lm=1/2/3/4 so eval can use smaller term maps without inventing a policy later.

## Expected metrics

This should reduce false-copy and over-adoption behavior while preserving a TERM_ACC advantage over no-RAG. The target is BLEU preservation relative to verified no-RAG, not maximum TERM_ACC.

## Verdict

Success. Built the De cap16 denoise/budget branch under `/mnt/gemini/data1/jiaxuanluo/speech_llm_de_cap16_denoise_budget_20260525/de/hn1024_tau078_cap16_denoise_budget_v1`.

Final train JSONL: `train_s_de_retriever_hn1024_tau078_cap16_denoise_budget_gttermwrap_exactboundary.jsonl`.
Final dev JSONL: `dev_s_de_retriever_hn1024_tau078_cap16_denoise_budget_gttermwrap_exactboundary_first355.jsonl`.

Validation passed with 12,500 train rows / 71,730 train chunks and 355 dev rows / 1,946 dev chunks. Train term-map chunk rate is 0.7926 with average 4.59 entries per chunk, down from the original dense cap16 retriever branch. All GT terms remain present in the rebuilt term maps (`gt_missing_from_termmap=0`), and assistant tags have `malformed_tag_messages=0` and `latin_boundary_cut_messages=0`.

The emitted runtime schedule is `runtime_termmap_budget_schedule.json`: lm1 max 6 terms, lm2 max 8, lm3/lm4 max 10, with `empty_term_map_policy=omit`.

Full `audit_training_jsonl.py` was skipped after it entered D-state I/O on the train file. The lightweight structural/tag/GT-in-termmap validation passed and is recorded in `validation_summary.json`.
