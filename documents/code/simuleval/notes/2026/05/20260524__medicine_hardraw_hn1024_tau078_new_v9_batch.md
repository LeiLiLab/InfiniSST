## Hypothesis

Batching all five medicine talks into one SimulEval process per latency
multiplier avoids repeated vLLM model loads and gives the raw hard-medicine
strict-term readout for the new-v9 speech LLM.

## Background / Motivation

The previous per-sample launcher loaded the same HF checkpoint once per
sample/lm pair.  For the current zh medicine raw-glossary main readout, the
five selected talks should be evaluated together for each lm.

## What changed vs baseline

- Strict raw medicine glossary comes from the LLM-judged manual hard-term file.
- Runtime retriever is HN1024 with tau 0.78 and top-k 10.
- Speech LLM is the exported new-v9 term-tag-delay/no-gt-zero old-new-v3 r32/a64
  HF checkpoint.
- The launcher builds combined five-talk source/target/source-text/ref/audio
  files and runs one process per lm.

## Expected metrics

Report BLEU, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL for lm 1, 2,
3, and 4 on the combined five-talk zh medicine set.

## Verdict

SUCCESS: zh lm=1,2,3,4 completed on the combined five-talk medicine set.
Each lm output directory has `instances.log` with 5 rows plus one-row
`eval_results.tsv` and `scores.tsv`; all detached lm PID files have exited, and
the `lm*.out` logs end with `[ALL DONE] medicine hard raw batch eval complete`.

Output root:
`/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T0242`
