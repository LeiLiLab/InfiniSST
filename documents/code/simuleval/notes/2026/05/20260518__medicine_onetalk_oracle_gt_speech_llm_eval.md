# Medicine One-Talk Oracle GT Term Map Speech LLM Eval

## Hypothesis

If the Speech LLM can use terminology evidence correctly, a sentence-aligned all-GT `term_map` should improve term adoption and TERM_ACC on the medicine one-talk readout without increasing sentence-level false copy rate.

## Background / Motivation

The retriever readout for `lh1b88kw` at tau 0.73 gives a realistic noisy term map.  Before exploring more Speech LLM SFT variants, we need an upper-bound check that bypasses retrieval and feeds only strict MFA-exact medicine GT terms with target translations from ESO v2 annotations.

## What changed vs baseline

- Baseline run URL: N/A for this first oracle upper-bound event.
- Diff: use `prepare_medicine_one_talk_inputs.py` to build a translated one-talk strict-GT glossary and `medicine.oracle_term_map__medicine_<sample>.json`; pass it to the SimulEval agent through `--oracle-term-map-path`; skip MaxSim retriever/index building.
- Scope: one ESO medicine talk, default `sample_404_v2`, default target language `zh`, default latency multiplier supplied at launch time.

## Expected metrics

Report BLEU, StreamLAAL, TERM_ACC, realAdopt, and sentence-level FCR.  Oracle GT should mainly test whether term evidence is usable by the Speech LLM; it is not a retriever comparison.

## Verdict

PENDING: run not launched yet because `MODEL_NAME_OVERRIDE` and `TARGET_LM` are launch-time choices.
