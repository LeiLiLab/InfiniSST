# Tagged ACL De lm4 Decode/Cache Probe

## Hypothesis

The de retriever-cap16 RASST model may be losing BLEU at lm=4 because decoding
or streaming-context settings are mismatched with the retrieved term maps.

## Background / Motivation

The verified de cap16 RASST curve has strong terminology accuracy but lm4 BLEU
remains below the verified InfiniSST/no-RAG BLEU gate. Offline systems gain
large BLEU from oracle terms, so we need to isolate whether the streaming gap is
caused by max-new-token limits, cached-history length, or stochastic decoding.

## What changed vs baseline

Keep model, retriever, glossary, tau, lm, and empty-term-map policy fixed:

- model: de retriever-cap16 exact-boundary SLM
- retriever: HN1024
- tau: 0.78
- lm: 4
- dataset: tagged ACL raw de
- empty term map policy: omit

Sweep only decode/cache settings:

- max_new_tokens
- max/keep cache seconds and chunks
- one greedy decoding diagnostic

## Expected metrics

Target: find any configuration with lm4 BLEU greater than 33 while preserving
TERM_ACC substantially above no-RAG.

## Verdict

Pending. Each config writes `summary_de_lm4.tsv` and `length_ratios_de_lm4.json`.
