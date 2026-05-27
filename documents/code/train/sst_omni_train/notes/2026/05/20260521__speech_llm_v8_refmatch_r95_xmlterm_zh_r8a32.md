# Speech LLM V8 refmatch R95 XML-term SFT: zh r8/a32

## Hypothesis

Wrapping term-map entries with `<term>...</term>` should increase term salience
relative to V7 plain `source=target` formatting, while preserving the same
reference-compatible R95 GT/noise curriculum.

## Background / Motivation

V7 fixes the main data-quality issue by only trusting source terms whose target
translation exactly appears in the reference.  It also keeps GT term-map recall
near the deployed retriever's expected 95% regime.  Prior tagged experiments had
small positive signs, so this run isolates a cleaner XML-style tag format on top
of V7.

## What changed vs baseline

- Baseline data: V7 refmatch R95 plain term-map data.
- New data: V8 refmatch R95 XML-term data.
- Term-map format: `<term>source => target</term>`.
- LoRA: rank 8, alpha 32.
- Compute: aries, two GPUs.
- Data and sampling distribution should match V7 except for term-map rendering.

## Expected metrics

Primary downstream check is tagged ACL `zh lm2/raw`.  V8 should be compared
directly against V7 to test whether XML tags improve exact TERM_ACC without
hurting BLEU or increasing false-copy rate.

## Verdict

Pending.
