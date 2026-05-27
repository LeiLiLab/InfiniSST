## Hypothesis

V9 dense top10 no-tau retriever-SFT and V10 marker-augmented SFT should improve over V7/V8 on tagged ACL zh lm2/raw.  V10 specifically tests whether stronger target-string copy supervision improves REAL_TERM_ADOPT.

## Background / Motivation

Old `new_v3` checkpoints beat V7/V8, suggesting the training distribution should be dense and capped rather than sparse/refmatch-clean.  V9/V10 are trained from the initial MCore model with rank 8 alpha 32 and evaluated with the deployed `lh1b88kw` tau=0.73 plain term_map inference pipeline.

## What changed vs baseline

- Eval setting: tagged ACL, `lang=zh`, `lm=2`, raw glossary.
- Retriever at inference: `lh1b88kw`, tau=0.73, timeline lookback 1.92s.
- V9 model: dense top10 no-tau SFT with exact-reference GT backfill.
- V10 model: V9 plus marker-augmented GT target translations during SFT.
- Inference term_map format: plain for both.

## Expected metrics

Primary metric is TERM_ACC.  Secondary metrics are REAL_TERM_ADOPT, TERM_FCR, BLEU, and StreamLAAL.  V9 should beat V7/V8; V10 is useful only if marker training transfers without producing output-distribution mismatch.

## Verdict

Pending.
