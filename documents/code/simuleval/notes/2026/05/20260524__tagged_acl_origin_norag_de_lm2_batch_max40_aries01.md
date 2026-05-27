## Hypothesis

The lower verified En-De `lm=2` InfiniSST/no-RAG tagged-ACL raw BLEU may be
affected by the `max_new_tokens=80` decoding cap. This run repeats the same
batch setup with `max_new_tokens=40`.

## Background / Motivation

The current same-lm batch no-RAG rerun produced BLEU below the older reusable
baseline row. Before treating the older row as stale, this run isolates the
decode-length cap while keeping the speech LLM, inputs, glossary, and no-RAG
configuration fixed.

## What changed vs baseline

Only `max_new_tokens` changes from 80 to 40. The run uses
`/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4` with
`DISABLE_RAG_OVERRIDE=1`, tagged ACL raw glossary, `de`, `lm=2`, same-lm batch
over five ACL talks, and assistant `<term>` tag stripping before metrics.

## Expected metrics

Compare BLEU, StreamLAAL, and TERM_ACC against the max80 no-RAG rerun and the
older TSV row for `acl_tagged_raw / InfiniSST / de / lm=2`.

## Verdict

Completed on Aries GPU 0,1.

- BLEU: 29.7466
- StreamLAAL: 1577.0870
- StreamLAAL_CA: 1007.6402
- TERM_ACC: 0.6417 (600 / 935)

This does not support `max_new_tokens=80` as the cause of the lower verified
no-RAG BLEU: reducing the per-update cap to 40 slightly lowered BLEU relative
to the max80 rerun (30.0676 -> 29.7466) while TERM_ACC moved only slightly
(0.6364 -> 0.6417).

The W&B wrapper again marked the run as failed because its final TSV discovery
missed the already-written `eval_results.tsv`; the run summary and manifest
were corrected after verifying the TSV, `instances.log`, and
`instances.strip_term.log`.

Strip validation passed: raw and stripped logs both have 5 rows, source and
reference fields are unchanged, no `<term>` tags remain in the scored
`instances.strip_term.log`, and German word-level delay counts match the
stripped predictions.
