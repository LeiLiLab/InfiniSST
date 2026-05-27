## Hypothesis

The prompt-supplied En-De `lm=2` InfiniSST/no-RAG tagged-ACL raw baseline may
need verification under the current same-lm batch harness and max-new-token
setting.

## Background / Motivation

The clean de RASST tagged-ACL raw result improves TERM_ACC but has lower BLEU
than the existing InfiniSST baseline table. This run remeasures the direct
InfiniSST configuration: original En-De speech LLM with no retriever.

## What changed vs baseline

This run uses `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`
with `DISABLE_RAG_OVERRIDE=1`, tagged ACL raw glossary, `de`, `lm=2`,
same-lm batch over five ACL talks, and `max_new_tokens=80`.

## Expected metrics

The key comparison is BLEU and TERM_ACC against the existing TSV row for
`acl_tagged_raw / InfiniSST / de / lm=2` and against the clean de RASST
`lm=2` row.

## Verdict

Completed on Aries GPU 0,1. The eval TSV was written successfully:

- BLEU: 30.0676
- StreamLAAL: 1541.8293
- StreamLAAL_CA: 1264.7621
- TERM_ACC: 0.6364 (595 / 935)

The W&B wrapper initially marked the run as failed because its final TSV
discovery missed the already-written `eval_results.tsv`; the run summary and
manifest were corrected after verifying the TSV, `instances.log`, and
`instances.strip_term.log`.

Strip validation passed: raw and stripped logs both have 5 rows, source and
reference fields are unchanged, no `<term>` tags remain in the scored
`instances.strip_term.log`, and German word-level delay counts match the
stripped predictions.
