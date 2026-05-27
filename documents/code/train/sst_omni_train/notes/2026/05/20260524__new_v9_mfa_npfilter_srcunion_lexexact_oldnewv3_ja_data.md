## Hypothesis

The ja Speech LLM SFT data should follow the same corrected line as de: merge wiki100k with old-new_v3 GigaSpeech source candidates, require MFA source exact support, and keep only target spans that are exactly supported by future assistant text.

## Background / Motivation

The pure wiki100k source glossary under-covered GigaSpeech training terms. The corrected de pipeline showed that source-union candidate coverage restores GT density. This ja run applies the same fix and avoids GT derivation from legacy term_map matches.

## What changed vs baseline

- Source glossary is wiki100k plus ja old-new_v3 noun/entity source candidates.
- Legacy term_map is used only as a translation lexicon and exact future-span filter.
- Stage A skips per-candidate OpenAI rewrite and keeps the exact supported target span.
- Downstream old-new_v3 TCM retriever term_map, GT backfill, no-GT-zero, and assistant `<term>` tags remain unchanged.
- Candidate extraction and retriever generation use 4-way sharding for speed.

## Expected metrics

Data-prep should produce substantially more GT terms than the pure wiki100k ja attempt while avoiding the fuzzy/term_map-derived contamination seen in rejected de/ja New V9 runs.

## Verdict

Pending data-prep completion and validation.
