# Tagged ACL De TM-SFT GT-Wrap lm4 Short-Cache Probe

## Hypothesis
The German TM-SFT exact GT-term-wrap SLM may preserve BLEU better than the cap16 SLM when paired with the newly selected short-cache decode setting, while retaining high terminology accuracy.

## Background / Motivation
The verified default-cache TM-SFT GT-wrap lm4 readout reached BLEU 32.2681 and TERM_ACC 0.8727, below the verified no-RAG lm4 BLEU target.  The cap16 SLM recovered BLEU under a shorter cache setting (max/keep seconds 40/20, max/keep chunks 8/4), so this readout tests whether the same inference setting helps the TM-SFT GT-wrap model.

## What changed vs baseline
Compared with `20260525T0413__simuleval__tagged_acl_tmsft_gttermwrap_exact_de_lm1to4_hn1024_tau078_batch_aries8`, only inference cache and empty term-map handling are changed:
- `lm=4` only.
- `max_new_tokens=80`.
- `max_cache_seconds=40`, `keep_cache_seconds=20`.
- `max_cache_chunks=8`, `keep_cache_chunks=4`.
- `empty_term_map_policy=omit`.
- HN1024 retriever, tau 0.78, tagged ACL raw glossary unchanged.

## Expected metrics
Primary gate: BLEU should exceed the verified no-RAG lm4 baseline 33.3008.  TERM_ACC should remain around the earlier TM-SFT GT-wrap level, ideally at or above 0.86.

## Verdict
Completed on Taurus GPU 3,7.

Result: BLEU 32.6166, StreamLAAL 2596.2700, StreamLAAL_CA 684.9767, TERM_ACC 0.8749 (818/935), REAL_TERM_ADOPT 0.896856, TERM_FCR 0.163728.

This improves over the earlier default-cache TM-SFT GT-wrap lm4 BLEU 32.2681, but it still fails the verified no-RAG BLEU gate 33.3008 and is below the cap16 short-cache readout BLEU 33.4820.  Therefore, the short-cache setting helps but the TM-SFT GT-wrap r32/a32 SLM is not the stronger lm4 candidate under this setup.
