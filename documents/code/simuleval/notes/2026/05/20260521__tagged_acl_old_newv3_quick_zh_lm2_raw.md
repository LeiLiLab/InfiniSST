## Hypothesis

Older `new_v3` Speech LLM term-map SFT checkpoints may outperform V7/V8 because their training term maps look closer to the legacy LLM-generated setup: denser maps, hard cap around 20 entries, and GT backfill rather than refmatch-clean sparse maps.

## Background / Motivation

V7/V8 refmatch-clean SFT improved over the broken V3 retriever-SFT line but still lost TERM_ACC to the no-TM-SFT and LLM-generated term-map SFT baselines on tagged ACL zh lm=2 raw. The old `new_v3` runs used a retriever-produced TCM/wiki100k term-map dataset with `tau=0.75`, `d9`, `k20`, post-filter cap, and GT zh override. The random variant preserved GT terms while thinning retriever-only negatives to reduce density.

## What changed vs baseline

- Eval setting: tagged ACL, `lang=zh`, `lm=2`, `glossary=raw`.
- Retriever at inference: current `lh1b88kw`, tau=0.73, timeline lookback=1.92s.
- Models:
  - `old_newv3_r32a64`: `/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_r32a64_taurus8/keep1.0_r32/v0-20260508-122348-hf`
  - `old_newv3_random_r32a64`: `/mnt/gemini/data2/jiaxuanluo/speech_llm_tcmw100kgt_tau075_new_v3_random_r32a64_aries8/keep1.0_r32/v1-20260508-123645-hf`
- Term-map serialization: plain.

## Expected metrics

Primary quick-check metric is TERM_ACC. Secondary metrics are BLEU, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL. If dense/capped retriever-style SFT is the missing ingredient, at least one old model should beat V7 TERM_ACC while keeping TERM_FCR materially below no-TM-SFT.

## Verdict

Completed on Aries GPUs 6,7.

| model | W&B | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| no TM-SFT baseline | prior tagged ACL readout | 45.30 | 87.19 | 88.87 | 22.08 | 1560 |
| V7 refmatch r95 plain | `mql4cqax` | 46.94 | 84.38 | 84.40 | 8.57 | 1679 |
| V8 refmatch r95 xml-tagged | `ok1qp7dk` | 47.00 | 82.70 | 82.45 | 9.09 | 1729 |
| old `new_v3` r32/a64 | `538mqu47` | 48.90 | 86.18 | 86.08 | 7.01 | 1777 |
| old `new_v3_random` r32/a64 | `u3v3lii2` | 49.34 | 85.96 | 85.84 | 8.57 | 1817 |

The old dense/capped retriever-style SFT checkpoints recover most of the no-TM-SFT TERM_ACC while cutting false-copy rate sharply. They also substantially outperform V7/V8 on BLEU and TERM_ACC in this quick setting.

Interpretation: V7/V8 likely over-cleaned the training term-map distribution. Requiring exact reference-match GT plus aiming for high GT-in-map rate made supervision too sparse/too oracle-like relative to inference, and XML tagging did not compensate. The older `new_v3` data is closer to the useful regime: dense maps, cap around 20 entries, GT backfill, and enough distractors to teach selective use without the fully broken high-noise V3 behavior.

Next candidate recipe: rebuild a retriever-SFT dataset that mimics old `new_v3` density rather than tau-filtered sparse maps: retrieve top-K without tau filtering, cap at 20, always backfill exact-reference GT terms, and keep a controlled random-thinning variant for negatives.
