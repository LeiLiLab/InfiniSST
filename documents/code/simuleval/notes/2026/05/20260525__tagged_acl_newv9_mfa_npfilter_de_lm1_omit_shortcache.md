## Hypothesis

For En-De tagged ACL lm=1, the stronger NewV9 MFA npfilter lexexact SLM may recover BLEU if empty retrievals are omitted instead of rendered as `term_map: NONE`, while using the same short-cache setting that helped recent de probes.

## Background / Motivation

The best verified RASST de/lm=1 BLEU so far is 26.2487 from NewV9 MFA npfilter lexexact + HN1024 tau=0.78, but that run used `term_map: NONE` blocks for empty retrievals. Later diagnostics showed that empty-map prompt shape and dense/noisy local term maps affect SLM behavior. The user requested a direct omit-empty readout with the current low-cache setting.

## What changed vs baseline

- Parent readout: `20260524T1738__simuleval__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_de_lm1to4_max80`.
- SLM unchanged: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_r32a64_tp2_taurus8/keep1.0_r32/v0-20260525-001708-hf`.
- Runtime retriever unchanged: HN1024, `tau=0.78`, `top_k=10`.
- Eval split unchanged: tagged ACL raw En-De, five talks.
- Empty term-map policy changed to `omit`.
- Cache changed to `max_cache_seconds=40`, `keep_cache_seconds=20`, `max_cache_chunks=8`, `keep_cache_chunks=4`.
- Decode cap kept at `max_new_tokens=80` to isolate omit/cache behavior from low cap truncation.

## Expected metrics

The key gate is BLEU relative to the previous de/lm=1 RASST best of 26.2487 and the tagged ACL raw InfiniSST baseline reference of 27.4672. TERM_ACC should stay near the prior RASST range if omission mainly reduces empty-prompt instability.

## Verdict

Completed. The omit-empty + short-cache inference-side ablation did not improve lm=1 BLEU for this NewV9 SLM.

| setting | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | TERM |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior `term_map: NONE` block, cache 80/60, chunks 16/8 | 26.2487 | 1015.7327 | 1302.5656 | 0.8428 | 788/935 |
| omit empty map, cache 40/20, chunks 8/4 | 25.6060 | 1020.6399 | 1023.8905 | 0.8471 | 792/935 |

Interpretation: this model was trained with `term_map: NONE` blocks, so simply omitting empty maps at inference is out-of-distribution and does not fix lm=1 BLEU. The result supports repairing the SFT data distribution rather than applying omit-only inference changes to this checkpoint.
