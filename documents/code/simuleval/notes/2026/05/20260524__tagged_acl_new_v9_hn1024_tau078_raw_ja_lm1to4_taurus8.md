# Tagged ACL ja raw New V9 HN1024 tau0.78 on taurus

## Hypothesis

The ja New V9 assistant-term-tag-delay Speech LLM should improve term adoption stability on tagged ACL raw glossary settings when paired with the HN1024 retriever at tau=0.78.

## Background / Motivation

Earlier ja tagged ACL runs showed sensitivity to glossary noise and low-latency cascades.  New V9 is the current main-line ja Speech LLM candidate trained with assistant-side term tagging and delayed boundary-term handling.

## What changed vs baseline

Runs `lm=1,2,3,4` in parallel on taurus, one 2-GPU pair per latency setting.

- Speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_ja_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-123624-hf`
- Retriever: HN1024 `lh1b88kw`
- Tau: `0.78`
- Runtime glossary: tagged ACL raw
- Metric denominator: fixed tagged ACL raw
- Output cleanup: strip `<term>` / `</term>` before scoring
- Launcher: `documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_taurus8.sh`

This replaces the earlier aries auto-watcher because taurus became fully idle.

## Expected metrics

Primary readouts are BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, and StreamLAAL for ja tagged ACL raw glossary at lm=1,2,3,4.  Metrics should be compared against the previous ja raw tagged ACL baselines under the same retriever/tau and fixed raw denominator policy.

## Verdict

Completed on taurus.  The launcher produced all four ja/raw `lm=1,2,3,4` eval files.  `lm=3` finished before this note was expanded and its first W&B logger call failed on missing note sections, so it was manually backfilled to W&B run `ycso1ibu`.

Summary:

| lm | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL | W&B |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 21.63 | 81.70 | 85.77 | 27.04 | 1496 | pqg9tvhd |
| 2 | 31.15 | 86.70 | 89.36 | 21.10 | 2196 | 5ht34lfe |
| 3 | 33.25 | 86.38 | 89.81 | 20.18 | 2751 | ycso1ibu |
| 4 | 33.24 | 87.66 | 91.46 | 16.80 | 3286 | 408pmg2v |

Summary artifact: `/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_ja_lm1to4_20260524T0933_tagacl_newv9_hn1024_tau078_raw_ja_taurus8/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/__summary__/summary_ja_raw_lm1to4.md`
