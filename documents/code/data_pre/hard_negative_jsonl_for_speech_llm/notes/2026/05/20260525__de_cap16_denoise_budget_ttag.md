# de Cap16 Denoise-Budget Short-Tag Data

## Hypothesis
Replacing assistant-side `<term>...</term>` supervision with shorter `<t>...</t>` markers should reduce token overhead and XML-style formatting pollution while preserving the same denoising-budget term-map exposure.

## Background / Motivation
The current de cap16 denoising-budget branch already passed lightweight validation and is the intended next SLM repair direction.  Its remaining risk is that long supervision tags increase generation budget pressure, especially at low latency multipliers.  This branch keeps the exact same term-map and GT-target wrapping policy but changes only the output marker shape.

## What changed vs baseline
- Parent data event: `20260525T1210__data_prepare__de_cap16_denoise_budget`.
- Inputs are the parent stage1 JSONLs before assistant target wrapping.
- Term maps, no-GT chunk policy, score dropout, and row/chunk membership are unchanged.
- Assistant target marker changes from `<term>{translation}</term>` to `<t>{translation}</t>`.
- Eval must use `--strip-output-tags term_t` so both legacy and short tags are removed before BLEU/latency scoring.

## Expected metrics
The target is not higher TERM_ACC at any cost.  The gate is BLEU recovery relative to the verified de InfiniSST/no-RAG baseline while preserving a clear TERM_ACC advantage.  Expected first readout: tagged ACL raw, de, HN1024, tau calibrated from dev, lm=4 and lm=1..3 with proportional max-new-token caps.

## Verdict
Pending.  Data generation and validation must complete before training is allowed to start.
