## Hypothesis

The ja New V9 speech LLM should improve hard-medicine strict-term adoption when paired with the HN1024 retriever at tau 0.78 for lm 1 and lm 2.

## Background / Motivation

This is the ja counterpart of the de/zh New V9 hard-medicine raw strict-glossary readout.  The first aries submission only runs lm 1 and lm 2 because GPUs 2,3,4,5 are available.

## What changed vs baseline

- Speech LLM: ja New V9 assistant-term-tag delay/no-gt-zero old-new-v3 r32/a64 HF export.
- Retriever: HN1024 `lh1b88kw`, tau 0.78, top-k 10, timeline mode with 1.92s lookback.
- Glossary: `/mnt/gemini/data1/jiaxuanluo/medicine_hard_manual_glossary_streamlaal_20260524.json`.
- Evaluation: lm 1 and 2; five samples per lm; output-side `<term>` tags stripped before scoring.

## Expected metrics

Report BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL for lm 1 and lm 2 on the combined five-sample ja medicine set.

## Verdict

Terminated on 2026-05-24.  Do not use these outputs as main results.  The ja/de New V9 SFT data path was found dirty: gt terms were derived from term-map matches rather than source-aligned chunks, and the assistant-side wrap step used an over-broad local rewrite fallback that can introduce noisy tags.  The active lm1/lm2 aries processes were stopped before completion.
