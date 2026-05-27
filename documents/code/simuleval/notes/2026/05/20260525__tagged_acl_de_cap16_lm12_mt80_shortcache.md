## Hypothesis

For the de retriever-cap16 exact-boundary SLM, fixing `max_new_tokens=80` for low-latency lm=1 and lm=2 may recover BLEU relative to the proportional decode caps lm1=20 and lm2=40.

## Background / Motivation

The selected short-cache cap16 readout achieved BLEU 33.4820 at lm=4 with `max_new_tokens=80`, but lm=1 and lm=2 used proportional caps and remained weak, especially lm=1. This probe isolates the decode-cap effect while keeping model, retriever, glossary, cache, and empty-term-map policy fixed.

## What changed vs baseline

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_retriever_cap16_exactboundary_r32a32_ep4_taurus8/keep1.0_r32/v1-20260525-141908-hf`.
- Runtime retriever: HN1024, `tau=0.78`, `top_k=10`.
- Eval dataset: tagged ACL raw En-De, five talks.
- Runtime cache: `max_cache_seconds=40`, `keep_cache_seconds=20`, `max_cache_chunks=8`, `keep_cache_chunks=4`.
- Empty term-map policy: `omit`.
- Decode cap: fixed `max_new_tokens=80` for both lm=1 and lm=2.

## Expected metrics

If BLEU loss is partly due to excessive truncation after adding `<term>` tags, lm=1 and lm=2 should improve without major TERM_ACC loss. If the issue is noisy term-map exposure or SLM streaming behavior, fixed cap 80 will not close the BLEU gap.

## Verdict

Completed with validated five-row `eval_results.tsv`, `instances.log`, and `instances.strip_term.log` for both lm=1 and lm=2.

Fixed `max_new_tokens=80` did not repair low-latency BLEU. lm=1 reached BLEU 23.6478 / TERM_ACC 0.8321, while the proportional-cap short-cache row was BLEU 23.4105 / TERM_ACC 0.7968. lm=2 reached BLEU 30.4212 / TERM_ACC 0.8503, while the proportional-cap row was BLEU 30.2953 / TERM_ACC 0.8503. The small BLEU movement indicates that truncation from lm1=20 or lm2=40 was not the primary failure mode.
