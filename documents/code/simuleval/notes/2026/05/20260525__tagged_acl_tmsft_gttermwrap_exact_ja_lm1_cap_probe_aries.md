# JA lm=1 Tagged Cap Probe

## Hypothesis
The exact GT-term-wrapped JA TM-SFT model can recover a usable `lm=1` tagged-ACL
RASST row if empty retrieval prompts are omitted, runtime term maps are less
aggressive, and `max_new_tokens` is capped below the previous fixed 80 setting.

## Background / Motivation
The previous JA `lm=1` tagged-ACL RASST row over-generated badly even after
switching empty retrieved term maps from explicit `term_map: NONE` to omitted
term-map blocks. Training data contains many 0.96s chunks, but those chunks
usually have empty or very short assistant targets, while the failed eval kept
generating long continuations over hundreds of stream steps.

## What changed vs baseline
This probe keeps the same JA exact GT-term-wrapped TM-SFT checkpoint, HN1024
retriever, tagged ACL raw glossary, and `lm=1`, but sweeps `max_new_tokens` while
using `empty_term_map_policy=omit`, `RAG_TOP_K=3`, and `tau=0.79`.

## Expected metrics
The first acceptable setting should keep all five instances complete, avoid long
repetition tails, keep maximum hypothesis/reference character ratio near or
below 1.4, and retain a clear term-accuracy gain over no-RAG.

## Verdict
Pending.
