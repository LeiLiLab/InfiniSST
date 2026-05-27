## Hypothesis

The previous no-RAG batch baseline was not prompt-equivalent to InfiniSST
serial no-RAG because the batch driver still used a term-map-aware system prompt
and, by default, an empty `term_map:NONE` block. Re-running the de/lm=1..4
baseline with batch evaluation, omitted empty term maps, and a serial-compatible
no-RAG system prompt should provide the correct batch gate for RASST comparisons.

## Background / Motivation

The earlier serial no-RAG lm=4 gate reported BLEU `33.3008`, while a batch
lm=4 no-RAG probe with `max_new_tokens=40` was much lower. A follow-up diagnosis
showed that the batch driver was not prompt-equivalent: the serial InfiniSST
no-RAG agent does not mention `term_map` in the system prompt and does not add
`term_map:NONE` when retrieval is disabled.

## What changed vs baseline

This event uses the batch vLLM evaluator for En-De tagged ACL raw no-RAG:

- `DISABLE_RAG=1`
- `empty_term_map_policy=omit`
- `norag_prompt_policy=serial_compat`
- `max_new_tokens=20*LM`: lm1=20, lm2=40, lm3=60, lm4=80
- five samples per LM in the same-LM batch
- effective completed retry runs all lm1..lm4 on Aries only, using GPU pairs
  4,5 and 6,7; the Taurus waiting queue was canceled before starting no-RAG
  evaluation.

The batch evaluator was extended with an explicit `--norag-prompt-policy` flag
so default batch/RAG behavior remains unchanged.

## Expected metrics

This is a verification readout, not a new model or calibration choice. The main
question is whether the batch no-RAG gate moves closer to serial InfiniSST when
the prompt policy is fixed. TERM_TOTAL should remain `935` for all LMs.

## Verdict

Completed on Aries only. The initial concurrent Taurus/Aries attempt failed
during vLLM KV-cache initialization with the known `ShmRingBuffer` shared-memory
race. The retry used Aries GPU pairs 4,5 and 6,7 and produced verified summaries
for all four latency multipliers.

Merged summary:
`/mnt/gemini/data1/jiaxuanluo/reports/20260525_de_origin_norag_batch_infinisst_prompt_max20lm_summary.tsv`

| lm | max_new_tokens | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 26.5639 | 1049.2446 | 1606.3982 | 0.6342 |
| 2 | 40 | 29.5552 | 1746.2115 | 1561.8990 | 0.6642 |
| 3 | 60 | 29.5526 | 2221.4341 | 1292.2720 | 0.6385 |
| 4 | 80 | 30.6587 | 2721.0152 | 964.8040 | 0.6492 |

Validation: every LM has one `eval_results.tsv`, five `instances.log` rows,
five `instances.strip_term.log` rows, and `TERM_TOTAL=935`.
