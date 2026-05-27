## Hypothesis

Rerunning the medicine no-RAG InfiniSST baseline for `de`, `lm=4` on aries
GPU 6,7 should produce the missing hard-manual StreamLAAL/TERM row for the
main-result table.

## Background / Motivation

The previous `de`, `lm=4` baseline attempt on Taurus wrote only a failed
`timing.tsv` row and no usable `instances.log` or hard-manual eval TSV. The
failure occurred during vLLM engine initialization with a shared-memory cleanup
error, so the row must be regenerated rather than reused.

## What changed vs baseline

This rerun keeps the same restored ESO medicine input set, model, glossary, and
no-RAG baseline launcher, but pins execution to aries GPU 6,7. Runtime caches
are moved under `/mnt/gemini/data1`, and `TMPDIR` is kept short under
`/dev/shm` to avoid local-root and IPC-path failures. The launcher also runs the
hard-manual StreamLAAL/TERM post-eval after generation succeeds.

## Expected metrics

The run should emit five hypotheses for the medicine sample group and then write
`eval_results_streamlaal_term.hard_llm_manual_check.tsv` under the `lm=4`
setting directory. The metric values are not assumed before the eval artifact is
created.

## Verdict

Cancelled and superseded. This launcher used the serial SimulEval path with
`--max-new-tokens 40`. The corrected rerun uses the same-lm batch evaluator,
concurrent scheduling for the five medicine samples, and `max_new_tokens=80`.
