# TM-SFT Exact GT Term Wrapping Data, Ja

## Hypothesis

Applying the same exact assistant-side `<term>` wrapping used for the German TM-SFT branch to the historical Japanese TM-SFT training set can provide a fast tagged-term SLM candidate without changing retriever exposure or term-map inputs.

## Background / Motivation

This branch mirrors the current German treatment:

- source data: `/mnt/gemini/data1/jiaxuanluo/train_s_ja_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl`
- derive GT terms only from embedded user `term_map:` entries whose target translation appears exactly in current/future assistant text
- wrap exact assistant target occurrences as `<term>...</term>`
- keep user prompts and `term_map` unchanged

No fuzzy GT derivation, no local rewrite, no LLM variant augmentation, and no no-GT term-map zeroing are used.

## What changed vs baseline

Compared with historical Japanese TM-SFT data, assistant messages receive exact `<term>` tags around supported GT target translations.

## Expected metrics

The downstream first readout should use tagged ACL raw Japanese at `lm=2`, HN1024, `tau=0.79`, same-lm batch, `max_new_tokens=80`, matching the German rescue branch protocol.

## Verdict

Pending data build and validation.
