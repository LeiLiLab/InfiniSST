## Hypothesis

Adding the old-new_v3 noun/entity source-candidate gate before MFA/OpenAI GT construction should remove common verb/adverb GT pollution in de Speech LLM SFT data while preserving source-side MFA exact evidence.

## Background / Motivation

The previous clean-MFA de data had correct audio/MFA alignment but too many one-word/common GT terms. The old zh winning lineage started from old-new_v3-style noun/entity candidates, then used backfilled GT, capped term maps, no-GT-zero, and assistant term tags.

## What changed vs baseline

- Generate an utterance-level old-new_v3 noun/entity/proper-noun allowlist with `extract_ner_candidates_v4.py`.
- Use the allowlist only as a phrase/type filter.
- Keep GT evidence from MFA source exact matching plus OpenAI exact future-reference span rewrite.
- Reuse the New V9 old-new_v3 TCM term-map builder, no-GT-zero, and `<term>` assistant tagging.

## Expected metrics

Dataset diagnostics should show fewer GT terms than the previous de New V9 build, lower common-word pollution, zero malformed tags, and `gt_in_term_map_rate=1.0` after GT backfill. Downstream eval should recover BLEU while preserving term adoption gains.

## Verdict

Pending data-prep completion and validation.
