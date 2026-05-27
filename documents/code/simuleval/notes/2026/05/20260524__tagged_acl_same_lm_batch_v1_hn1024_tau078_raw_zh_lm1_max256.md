# Tagged ACL Same-LM Batch V1 zh lm1 raw max256

## Hypothesis

Increasing the fixed decode budget from 40 to 256 new tokens should reduce lm=1 generation truncation and repeated-token degeneration without changing retrieval, cache, or decoding distribution otherwise.

## Background / Motivation

The current zh lm=1 raw tagged ACL result has low TERM_ACC, and miss inspection shows several KinyaBERT sentences where the model degenerates into repeated English text such as `consistently...` despite the relevant terms being present in the aligned term map.  Because New V9 can emit assistant-side `<term>` tags, the old 40-token budget is likely too tight for lm=1.

## What changed vs baseline

- Same model, retriever, raw tagged ACL glossary, tau, cache policy, and sampling parameters as the same-lm batch validation.
- `lm=1` only.
- Fixed `max_new_tokens=256` instead of 40.
- Uses the exact same-lm batch path: batched Whisper feature extraction, per-sample Qwen audio encoder for serial-equivalent retrieval, and five ACL talks in one vLLM process.

## Expected metrics

TERM_ACC and BLEU should improve over the serial lm=1 raw result if the low score is mainly due to output budget or short-chunk degeneration.  StreamLAAL should be interpreted from simulated delay, not wall-clock.

## Verdict

Pending.
