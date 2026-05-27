# Tagged ACL V16 LLM-Variant R32 Quick Eval

## Hypothesis

Increasing V16 LLM-variant LoRA capacity from r8/a32 to r32/a64 may improve
term-map adoption on tagged ACL `zh lm=2 raw`.

## Background / Motivation

The r32/a64 TP2 training run `pqg310ag` finished and exported a complete HF
checkpoint.  This quick eval compares it against the earlier r8/a32 V16
LLM-variant readout using the same retriever and strict raw tagged denominator.

## What changed vs baseline

- Speech LLM: `speech-llm-v16-llmvariant-zh-r32a64-tp2-m4096-aries2_keep1.0_r32`.
- Model checkpoint: latest complete HF export under the r32/a64 save root.
- Eval: tagged ACL, `zh`, `lm=2`, runtime glossary `raw`.
- Retriever: `lh1b88kw`, top-10, tau=0.73, timeline lookback=1.92s.
- Metric denominator: fixed raw tagged glossary.

## Expected metrics

Report BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL.  A useful
capacity increase should improve TERM_ACC and REAL_TERM_ADOPT without a large
BLEU or latency regression.

## Verdict

Success.  Eval run `fo2p096d` completed on tagged ACL `zh lm=2 raw`.

Metrics: BLEU 48.20, TERM_ACC 84.04%, REAL_TERM_ADOPT 83.81%, TERM_FCR 6.49%,
and StreamLAAL 1818.08.  The larger r32/a64 V16 LLM-variant checkpoint did not
match the later New V5 no-GT-zero old-new_v3 result on term adoption.
