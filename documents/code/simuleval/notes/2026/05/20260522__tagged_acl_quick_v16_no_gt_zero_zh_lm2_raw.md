# Tagged ACL quick eval for V16 no-GT-zero

## Hypothesis

Zeroing term maps on no-GT chunks in the V16 LLM-variant SFT data may reduce false-positive conditioning and improve `REAL_ADOPT` / `TERM_ACC` on `zh lm=2 raw`.

## Background / Motivation

V16 improved over V15 on the quick `zh lm=2 raw` readout, but the training data still contained retriever terms for chunks without reference-matched GT terms.  This eval checks the no-GT-zero ablation using the same tagged ACL quick pipeline.

## What changed vs baseline

- Model: V16 LLM-variant augmentation with no-GT chunks set to `term_map:NONE`.
- Eval: tagged ACL, `lang=zh`, `lm=2`, `glossary=raw`.
- Inference format remains plain `source=target`.

## Expected metrics

Primary metrics are `TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.  A useful result should keep the V16 adoption gain while lowering false-copy behavior.

## Verdict

Success.  Quick `zh lm=2 raw` readout: BLEU 48.33, TERM_ACC 87.75%, REAL_TERM_ADOPT 89.10%, TERM_FCR 6.49%, StreamLAAL 1951.80.  This improves adoption over V16, with a latency increase.
