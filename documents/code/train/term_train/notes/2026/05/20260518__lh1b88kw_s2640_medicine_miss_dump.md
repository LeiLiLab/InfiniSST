# lh1b88kw s2640 medicine miss dump

## Hypothesis

The medicine recall gap is driven by a small set of recurring failure modes that should be visible in per-query miss cases: acoustically similar medicine terms, transcript/MFA alignment issues, or glossary-scale distractors that outrank the ground-truth medicine term.

## Background / Motivation

The direct eval run `mry7kesp` showed medicine recall materially below dev/ACL/tagged ACL for the same `lh1b88kw` checkpoint at step 2640. Aggregate recall does not explain whether the gap comes from term coverage, score margins, or confusing distractors.

## What changed vs baseline

No model or metric behavior changes. This analysis reruns medicine-only eval from the same checkpoint and writes per-sample miss dumps for `base`, `gs1000`, and `gs10000` banks using the same positive-mask definition as `eval_medicine/recall@10`.

## Expected metrics

The aggregate medicine recall values should reproduce the prior eval within deterministic eval tolerance. New artifacts should include JSONL and Markdown miss cases with the target term, positive-term rank, top-10 retrieved terms, score margin, transcript text, MFA timing, and audio path.

## Verdict

Completed in W&B run `6ytwzug3`. The run reproduced the prior medicine recall profile and wrote per-bank miss-case JSONL/Markdown artifacts under `/mnt/gemini/home/jiaxuanluo/analysis_outputs/direct_lh1b88kw_medicine_miss_gpu7_20260518T172557`.

The miss dump shows the drop is mostly not a random single-term issue. The same 446 examples miss in both base and gs10000, while 79 extra examples become misses only after adding the larger medicine glossary bank. Frequent hard misses include German/radiotherapy-planning terms and short oncology labels; the larger bank adds many high-similarity drug-like distractors.
