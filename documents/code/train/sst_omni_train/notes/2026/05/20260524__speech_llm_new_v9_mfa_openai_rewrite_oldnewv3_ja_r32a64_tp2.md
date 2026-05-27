## Hypothesis

Training ja on the clean MFA+OpenAI New V9 data should reduce noisy term_map over-adoption while improving exact term adoption versus the rejected polluted ja branch.

## Background / Motivation

The earlier ja New V9 data is rejected because GT came from term_map/fuzzy matching. This run uses the rebuilt clean data from `speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_ja_20260524`.

## What changed vs baseline

- Uses r32/a64, TP=2, EP=4, one epoch.
- Uses clean MFA source candidates, OpenAI exact span rewrite, old-new_v3 TCM term_map, no-GT-zero, and assistant `<term>` tags.
- Save path defaults to `/mnt/gemini/data1` because `/mnt/aries/data7` is full.

## Expected metrics

Quick eval should be run on tagged ACL raw glossary, HN1024 tau=0.78, lm=1..4, with `<term>` tags stripped before metrics.

## Verdict

Pending.
