# Medicine sample404 HN1024 tau0.78, New V9 assistant term-tag SFT

## Hypothesis

New V9 assistant-side `<term>...</term>` supervision may improve exact term adoption on medicine.  Evaluation must strip the output-side `<term>` markers before BLEU, TERM_ACC, REAL_ADOPT, TERM_FCR, and StreamLAAL scoring.

## Background / Motivation

Current medicine gs10k PR quick readout suggests HN1024 tau=0.78 is the stable main-line operating point.  The Speech LLM checkpoint is exported from `speech-llm-new_v9-termtag-delay-clean-oldnewv3-r32a64-tp2-taurus8_keep1.0_r32`.

## What changed vs baseline

- Speech LLM: New V9 assistant term-tag delay-clean SFT.
- Retriever: HN1024 `lh1b88kw` checkpoint.
- Threshold: tau=0.78.
- Domain: medicine, sample 404 only.
- Runtime glossary: medicine gs10k.
- Metric denominator: fixed strict raw medicine per-talk glossary from `prepare_medicine_one_talk_inputs.py`.
- Output cleanup: strip literal `<term>` and `</term>` from generated hypotheses before scoring.

## Expected metrics

Compare against the previous New V5 no-gt-zero oldnewv3 r32 sample404 readout around TERM_ACC 90.30 and REAL_ADOPT 93.30.  Watch BLEU and StreamLAAL for tag-related side effects.

## Verdict

Completed as W&B `4b86tuti`.  With output-side `<term>` tags stripped before scoring, New V9 reaches BLEU `44.61`, TERM_ACC `92.54`, REAL_ADOPT `93.78`, TERM_FCR `15.33`, SOURCE_SENT_FCR `13.74`, and StreamLAAL `2118.03` on medicine sample 404 with HN1024 tau `0.78` and gs10k runtime glossary.

This improves TERM_ACC over the previous New V5 sample404 reference around `90.30` while keeping REAL_ADOPT roughly comparable, but StreamLAAL is high and FCR remains non-trivial.
