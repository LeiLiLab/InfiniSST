## Hypothesis

The short-cache setting selected from the En-De lm=4 probe should also improve lower-latency readouts when the decode cap is scaled with the latency multiplier.

## Background / Motivation

The full five-talk En-De lm=4 readout with `max_cache_seconds=40`, `keep_cache_seconds=20`, `max_cache_chunks=8`, and `keep_cache_chunks=4` improved BLEU to 33.4820 while preserving high tagged-ACL TERM_ACC. The user asked to run lm=1,2,3 with proportionally lower `max_new_tokens`.

## What changed vs baseline

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_de_retriever_cap16_exactboundary_r32a32_ep4_taurus8/keep1.0_r32/v1-20260525-141908-hf`
- Training lineage: `wtkgnf8k`, German retriever HN1024 tau=0.78 cap16 exact-boundary gt-term wrapping.
- Eval dataset: tagged ACL raw En-De, five talks.
- Runtime retriever: HN1024, `tau=0.78`, `top_k=10`.
- Runtime cache: `max_cache_seconds=40`, `keep_cache_seconds=20`, `max_cache_chunks=8`, `keep_cache_chunks=4`.
- Decode caps: lm1=20, lm2=40, lm3=60.

## Expected metrics

BLEU should recover relative to the previous full lm1-3 cap16 readout while TERM_ACC remains substantially above no-RAG. lm1 is the highest-risk setting because short context and term-map noise have previously caused BLEU degradation.

## Verdict

Completed. All three standalone artifact-backed batched vLLM readouts finished with 5 raw rows and 5 strip-term rows each.

Results:

| lm | max_new_tokens | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 23.4105 | 1047.3306 | 995.1145 | 0.7968 |
| 2 | 40 | 30.2953 | 1641.6537 | 930.2462 | 0.8503 |
| 3 | 60 | 31.7997 | 2237.8186 | 1178.7785 | 0.8588 |

The selected lm=4 reference row from the prior short-cache probe is BLEU 33.4820 and TERM_ACC 0.8674 with max_new_tokens=80. The combined selected TSV is `documents/code/simuleval/reports/20260525_de_shortcache_lm1to4_selected.tsv`.
