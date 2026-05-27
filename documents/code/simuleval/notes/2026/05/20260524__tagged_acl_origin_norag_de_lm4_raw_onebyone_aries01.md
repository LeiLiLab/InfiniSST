## Hypothesis

The de `lm=2` InfiniSST/no-RAG reruns may fail to reproduce older values because
the newer batch-vLLM harness changes behavior. This run checks the older
SimulEval/no-RAG path with `lm=4` while forcing the five ACL talks to run one at
a time.

## Background / Motivation

An existing non-batch de `lm=4` no-RAG rerun already produced a higher BLEU
point than the new batch-vLLM `lm=2` probes. To make the batch-vs-serial
comparison sharper, this run splits the five ACL dev talks into five independent
single-sample SimulEval invocations, then concatenates the resulting
`instances.log` files in original order and performs one fixed raw tagged
post-eval.

## What changed vs baseline

The model is still `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`.
RAG remains disabled. The runtime uses the old no-RAG SimulEval baseline
launcher, `de`, `lm=4`, tagged ACL raw metric glossary, and `max_new_tokens=40`.
The difference is that each ACL sample is generated in its own SimulEval run.

## Expected metrics

Compare against the prior full-list serial `lm=4` no-RAG rerun:
BLEU 33.3008, StreamLAAL 2824.4372, StreamLAAL_CA 4100.5704, TERM_ACC 0.6909.

## Verdict

Pending.
