# Zero-shot oracle term_map readout: origin bsz4 baseline on medicine one-talk

## Hypothesis

The pure streaming zh baseline may already pay attention to explicit `term_map` text at inference time, even without term-map-specific SFT. A true all-GT oracle term map on one medicine talk tests this zero-shot behavior.

## Background / Motivation

Before training an all-GT Speech LLM, we need to know whether the existing baseline can use clean terminology evidence as prompt context. This avoids attributing gains to SFT when the model may already copy clean term maps.

## What changed vs baseline

- Speech LLM HF checkpoint:
  - `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-zh-s_origin-bsz4`
- Evaluation mode:
  - one ESO medicine talk
  - strict MFA-only translated GT terms
  - `--oracle-term-map-path` injection
- No retriever is used, and no SFT is performed.

## Expected metrics

If the baseline can use term maps zero-shot, `TERM_ACC` and `REAL_TERM_ADOPT` should improve relative to no-term-map baseline behavior, while sentence-level `TERM_FCR` should remain low because the term map is oracle/clean.

## Verdict

Completed. Exact metrics are recorded in WandB run `x818tsc6`; the readout shows the baseline can use oracle term maps zero-shot, while false-copy remains non-trivial.
