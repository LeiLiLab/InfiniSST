## Hypothesis

Run the remaining JA hard-medicine New V9 readout for lm=3 and lm=4 with the
same fixed raw medicine glossary and HN1024 tau=0.78 setting used by lm=1/2.

## Background / Motivation

JA lm=1/2 were already submitted on aries GPUs 2-5.  Aries GPUs 6/7 became
available, so lm=3 is launched immediately and lm=4 is queued by a detached
waiter that checks for an idle two-GPU pair every five minutes.

## What changed vs baseline

- Language: ja
- Latency multipliers: 3 and 4
- Samples: 404, 545006, 596001, 605000, 606
- Glossary: raw strict hard medicine glossary
- Retriever: HN1024, tau=0.78, timeline lookback 1.92s
- Speech LLM: New V9 JA r32/a64 HF export

## Expected metrics

This is a quick five-sample medicine readout.  TERM metrics use the fixed raw
hard-medicine glossary denominator and strip any `<term>` tags from generated
text before scoring.

## Verdict

Terminated on 2026-05-24.  Do not use these outputs as main results.  The ja/de New V9 SFT data path was found dirty: gt terms were derived from term-map matches rather than source-aligned chunks, and the assistant-side wrap step used an over-broad local rewrite fallback that can introduce noisy tags.  The active lm3 aries process was stopped before completion; the lm4 waiter should also be considered invalid if it emitted any partial output.
