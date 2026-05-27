# Tagged ACL de raw: New V9 + HN1024 tau0.78

## Hypothesis

New V9 term-tag-delay SFT should improve German tagged ACL term adoption under the fixed raw tagged ACL glossary setting when paired with the HN1024 retriever at tau 0.78.

## Background / Motivation

The main RASST setting uses retrieval-conditioned speech LLM decoding.  This run completes the German raw-glossary main-result sweep for latency multipliers 1 through 4 using the de New V9 speech LLM.

## What changed vs baseline

Compared with the no term-map SFT baseline, this run uses the New V9 speech LLM trained with assistant-side term tagging/delay cleanup.  The retriever is fixed to HN1024 `lh1b88kw`, tau `0.78`, top-k `10`, and timeline lookback `1.92s`.

## Expected metrics

Primary metrics are BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL.  TERM metrics use the fixed raw tagged ACL glossary denominator, independent of runtime glossary size.  Outputs strip `<term>...</term>` markers before metric computation.

## Verdict

Completed.  Fixed raw tagged ACL results:

| lm | BLEU | TERM_ACC | REAL_ADOPT | TERM_FCR | StreamLAAL |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 25.05 | 82.67 | 85.23 | 21.28 | 918 |
| 2 | 27.24 | 83.64 | 84.40 | 22.62 | 1501 |
| 3 | 30.07 | 83.74 | 84.64 | 18.51 | 2130 |
| 4 | 30.62 | 84.81 | 85.83 | 20.40 | 2460 |

Purpose: run German tagged ACL main-result readout for `lm=1,2,3,4` with the New V9 speech LLM and fixed raw tagged ACL glossary denominator.

Launcher:

```bash
documents/code/simuleval/launchers/2026/05/20260524__tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_taurus_auto.sh
```

Core setting:

- language: `de`
- latency multipliers: `1 2 3 4`
- runtime glossary: `raw`
- metric glossary: fixed raw tagged ACL glossary
- retriever: HN1024 `lh1b88kw`
- threshold: `tau=0.78`
- timeline lookback: `1.92s`
- speech LLM: `/mnt/gemini/data1/jiaxuanluo/slm/speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_de_r32a64_tp2_taurus4/keep1.0_r32/v0-20260524-121145-hf`
- output tag cleanup: strip `<term>...</term>` markers before metrics

The launcher polls `nvidia-smi` on taurus and starts one 2-GPU setting at a time when an idle pair is available.

Expected summary:

```bash
/mnt/gemini/data1/jiaxuanluo/tagged_acl_new_v9_hn1024_tau078_raw_de_lm1to4_${RUN_STAMP}/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/__summary__/summary_de_raw_lm1to4.md
```
