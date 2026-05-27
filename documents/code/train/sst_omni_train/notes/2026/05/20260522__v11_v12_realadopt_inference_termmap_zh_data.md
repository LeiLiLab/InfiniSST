## Hypothesis

V11/V12 should improve Speech LLM `REAL_TERM_ADOPT` by making SFT chunks match the deployed low-latency regime and by making positive term-map entries locally necessary for the assistant target.

## Background / Motivation

Current retriever recall is already around 95%+ in the relevant readouts, so the main loss is adoption: `TERM_ACC ~= recall * REAL_ADOPT`.  V9 used dense top10 no-tau term maps and 100% GT backfill, which made term maps too noisy and too always-present.  This data line instead uses deployed tau-filtered timeline retrieval and avoids GT backfill.

## What changed vs baseline

- Reshape SFT chunks to effective `lm=3..6`; drop original chunks with `lm>6`.
- For original `lm<3`, buffer consecutive chunks instead of padding retriever input.
- Run deployed timeline retriever with `top_k=10`, `tau=0.73`, `lookback=1.92s`.
- Define GT supervision only when the retrieved source term is in chunk GT and its target translation is an exact substring of the local assistant output.
- V11 keeps realistic tau-filtered term maps.
- V12 further marker-augments local exact GT target translations in both term_map and assistant output.

## Expected metrics

Data diagnostics should show:

- output chunk multipliers only in `3..6`;
- lower term-map density than V9;
- no 100% all-chunk term-map behavior;
- nonzero local exact GT signal;
- no dropped malformed rows except explicitly logged `lm>6` filtering.

Downstream quick eval target is tagged ACL `zh lm2 raw`, with primary metric `REAL_TERM_ADOPT` and secondary `TERM_ACC/BLEU/TERM_FCR`.

## Verdict

Running data preparation.  Training must wait until diagnostics confirm the local exact-GT signal is not too sparse.
