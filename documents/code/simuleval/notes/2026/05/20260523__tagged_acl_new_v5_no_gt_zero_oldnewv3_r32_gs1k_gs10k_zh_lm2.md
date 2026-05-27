# Tagged ACL New V5 no-GT-zero old-new_v3 R32 zh lm2 gs1k/gs10k

## Hypothesis

New V5 no-GT-zero old-new_v3 r32/a64 is the current zh `lm=2` raw winner.  This
readout checks whether the same model remains robust when the runtime glossary
expands to gs1k and gs10k.

## Background / Motivation

The recovered training checkpoint from `cg5qisu9` produced a strong raw quick
eval as W&B `342oxpmu`: BLEU 48.20, TERM_ACC 90.00%, REAL_ADOPT 90.19%,
TERM_FCR 7.53%, and StreamLAAL 1663.60.

The original training run is marked failed only because data7 filled during
TensorBoard logging after `iter_0001000` had already been saved.  The checkpoint
was successfully exported to HF on data6.

## What changed vs baseline

- Speech LLM: `speech-llm-new_v5-no-gt-zero-oldnewv3-r32a64-tp2-aries2_keep1.0_r32`.
- Eval: tagged ACL, `zh`, `lm=2`.
- Runtime glossary kinds: `gs1k gs10k`.
- Metric denominator: fixed raw tagged ACL glossary, not glossary-size-specific.
- Retriever: `lh1b88kw`, top-10, tau=0.73, timeline lookback=1.92s.

## Expected metrics

Compare against the raw readout `342oxpmu`.  Main metrics are BLEU, TERM_ACC,
REAL_ADOPT, TERM_FCR, and StreamLAAL.  A useful model should keep TERM_ACC and
REAL_ADOPT close to raw while avoiding a large BLEU drop under larger glossary
noise.

## Verdict

Pending eval.
