## Hypothesis

After aligning the training-data system prompt with the inference agent prompt,
the batched InfiniSST/no-RAG En-De lm=4 baseline may change relative to the
previous batch readout.

## Background / Motivation

The prior de/lm4 no-RAG batch baseline was used as the verified InfiniSST
comparison point. A prompt mismatch between SLM training data and the inference
agent was later corrected, so this run refreshes only the affected batch
baseline condition before using it as a paper-facing comparison.

## What changed vs baseline

- Reuse the existing tagged ACL raw En-De batch no-RAG launcher.
- Run only lm=4.
- Keep `DISABLE_RAG=1`, `RAG_TOP_K=0`, and `EMPTY_TERM_MAP_POLICY=omit`.
- Use the current `batched_vllm_rag_eval.py` prompt code with
  `NORAG_PROMPT_POLICY=serial_compat`.
- Keep `max_new_tokens=20*lm=80`.

## Expected metrics

The main check is whether BLEU changes from the previous de/lm4 no-RAG batch
readout. `instances.log` and `instances.strip_term.log` must each contain the
five tagged ACL talks.

## Verdict

Completed successfully on Taurus GPUs 0,1.

Summary:

| lang | lm | max_new_tokens | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| de | 4 | 80 | 30.6587 | 2721.0152 | 585.4564 | 0.6492 |

Validation:

- `eval_results.tsv` has one aggregate row.
- `instances.log` has 5 rows.
- `instances.strip_term.log` has 5 rows.
- The launcher wrote `.success` and `[ALL DONE]`.

The stderr Traceback entries are non-fatal vLLM shared-memory cleanup warnings
after output writing, not metric failures.
