# PSC zh Tagged ACL New V9 HN1024 Tau0.78 Raw

## Hypothesis

New V9 assistant-side term-tag SFT with HN1024 retrieval at tau `0.78` should
provide the current zh tagged ACL RASST main-result line under the fixed raw
tagged denominator.

## Background / Motivation

The existing tagged ACL main-result table has zh raw rows for older no-TM-SFT
and RASST systems.  The current Speech LLM candidate is New V9
`termtag_delay`, validated on medicine with output-side `<term>` tags stripped
before scoring.  PSC already has the tagged ACL data, HN1024 retriever
checkpoint, raw/gs glossaries, runtime env, FBK fairseq, and mwerSegmenter.

## What changed vs baseline

- Run only zh tagged ACL raw main-result settings: `lm=1 2 3 4`.
- Use Speech LLM HF export
  `speech_llm_new_v9_assistant_termtag_delay_clean_no_gt_zero_oldnewv3_zh_r32a64_tp2_taurus8`.
- Use HN1024 `lh1b88kw` checkpoint with `tau=0.78`, `top_k=10`, and timeline
  lookback `1.92s`.
- Keep TERM metrics fixed to
  `acl6060_tagged_gt_raw_min_norm2.json`; runtime glossary is also raw.
- Strip output-side `<term>` tags before BLEU, StreamLAAL, and term metrics.
- Run on PSC Bridges-2 4x V100-32 with TP=4 and Apptainer
  `ubuntu_22_04_gcc.sif`.

## Expected metrics

Four zh raw rows should be produced for `lm=1,2,3,4`, each with BLEU,
StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and SOURCE_TERM_SENT_FCR.  The
values should be W&B-logged under family `tagged_acl_new_v9_hn1024_tau078`.

## Verdict

Planned.  Upload the New V9 HF export to public HF, pull it to PSC, run a smoke
job first, then submit the four full zh raw jobs only if smoke generation and
post-eval both pass.
