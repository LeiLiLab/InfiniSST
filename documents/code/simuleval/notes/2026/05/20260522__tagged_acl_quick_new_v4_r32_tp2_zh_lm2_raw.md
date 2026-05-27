# Tagged ACL quick eval for New V4 r32/a64 TP2

## Hypothesis

New V4 applies LLM-variant target-translation augmentation on the older stronger `new_v3` data line.  The `r32/a64` TP2 retry should be evaluated on the same high-signal tagged ACL `zh lm=2 raw` setting.

## Background / Motivation

The first New V4 r32/a64 run failed with TP=1 OOM.  The TP2 retry finished and exported an HF checkpoint, so this eval checks the intended rank-32 model rather than the failed or cancelled runs.

## What changed vs baseline

- Model: `rf17uw7x`, New V4 LLM-variant augmentation on old `new_v3`, LoRA r32/a64, TP2.
- Eval: tagged ACL, `lang=zh`, `lm=2`, `glossary=raw`.
- Inference term-map format remains plain `source=target`.

## Expected metrics

Primary metrics are `TERM_ACC`, `REAL_ADOPT`, `TERM_FCR`, BLEU, and StreamLAAL.  This is compared against V16 and V16 no-GT-zero quick readouts.

## Verdict

Success.  Quick `zh lm=2 raw` readout: BLEU 48.33, TERM_ACC 88.99%, REAL_TERM_ADOPT 89.06%, TERM_FCR 9.35%, StreamLAAL 1705.82.  This improves TERM_ACC over V16 no-GT-zero with much lower latency, but has higher FCR.
