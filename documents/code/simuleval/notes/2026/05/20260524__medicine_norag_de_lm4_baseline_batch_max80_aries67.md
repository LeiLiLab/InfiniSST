## Hypothesis

The missing medicine no-RAG InfiniSST baseline row for `de`, `lm=4` should be
rerun with the same-lm batch evaluator so the five medicine samples advance
concurrently under one vLLM instance.

## Background / Motivation

The previous Taurus row failed before producing a usable hard-manual eval TSV.
The first aries rerun used the serial SimulEval path and `max_new_tokens=40`,
which does not match the desired batch setup. This rerun corrects both issues.

## What changed vs baseline

This launcher uses `20260524__batched_vllm_rag_eval.sh` with RAG explicitly
disabled, `scheduler_batch_size=5`, `max_num_seqs=5`, and
`max_new_tokens=80`. It keeps the origin `gigaspeech-de-s_origin-bsz4` model,
restored ESO medicine audio/reference inputs, and hard-manual medicine glossary
for offline scoring.

## Expected metrics

The run should produce five instance rows for `de`, `lm=4` and write
`eval_results_streamlaal_term.hard_llm_manual_check.tsv` from the hard-manual
glossary scoring pass.

## Verdict

Success with a post-run boundary-spacing correction. The batch run completed on
aries GPU 6/7 with five lm=4 medicine samples, RAG disabled, and
`max_new_tokens_policy=fixed {4: 80}`.

The original `instances.log` is not valid for German word-level StreamLAAL:
generated chunks were concatenated without adding a space at chunk boundaries,
so prediction word counts did not match delay counts. For example, instance 0
had 5642 whitespace tokens but 5916 delay entries. This caused mWER
sentence-level resegmentation to create many zero-delay segments and produced
negative StreamLAAL.

The invalid standard filenames were moved to `*.invalid_boundary_bug.*` audit
copies so downstream scripts do not accidentally read the negative-latency row.

The preferred metrics are from
`eval_results_streamlaal_term.hard_llm_manual_check.boundaryfix.tsv`, rebuilt
from runtime `llm_output` records with spaces inserted at German chunk
boundaries:

`BLEU=27.81957520314305`, `StreamLAAL=2827.5546192535794`,
`StreamLAAL_CA=826.5603601106437`, `TERM_ACC=0.4735`, `TERM_CORRECT=340`,
`TERM_TOTAL=718`.

The invalid original TSV row was:

`BLEU=27.538561959014395`, `StreamLAAL=-3538.119780461579`,
`StreamLAAL_CA=-3599.416813837647`, `TERM_ACC=0.4805`, `TERM_CORRECT=345`,
`TERM_TOTAL=718`.

Completion evidence: `instances.log` has 5 rows, `instances.strip_term.log` has
5 rows, the wrapper wrote `[ALL DONE] de lm4 no-RAG batch max80`, and aries GPU
6/7 memory returned to 2 MiB.
