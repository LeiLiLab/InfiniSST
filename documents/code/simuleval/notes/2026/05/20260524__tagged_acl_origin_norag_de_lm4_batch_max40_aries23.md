# Tagged ACL origin no-RAG de lm4 batch max40

## Hypothesis

The En-De `lm=4` InfiniSST/no-RAG main-result point should be checked with the
same-lm batch evaluator to separate an evaluator effect from the existing serial
SimulEval baseline.

## Background / Motivation

The current main-result `acl_tagged_raw / InfiniSST / de / lm=4` point comes
from the verified serial no-RAG rerun. Earlier `lm=2` no-RAG batch checks showed
that the batch harness is close but not identical to the serial path, so `lm=4`
needs a direct batch readout before using it for interpretation.

## What changed vs baseline

This run uses `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`
with `DISABLE_RAG_OVERRIDE=1`, tagged ACL raw glossary, `de`, `lm=4`,
same-lm batch over five ACL talks, and `max_new_tokens=40`. The `max_new_tokens`
value matches the verified serial lm4 main-result rerun.

## Expected metrics

Compare BLEU, StreamLAAL, and TERM_ACC against the verified serial
`20260524T160830` InfiniSST/no-RAG lm4 row and the existing lm2 batch
diagnostics.

## Verdict

Completed on Aries GPU `2,3`. The batch evaluator wrote the TSV/log artifacts:

- BLEU: 30.0243
- StreamLAAL: 2675.3594
- StreamLAAL_CA: 882.8662
- TERM_ACC: 0.6738 (630 / 935)
- W&B run: `ixmu9jhv`

The W&B wrapper initially marked the run failed because its final TSV discovery
missed the already-written `eval_results.tsv`. The run was corrected after
directly validating `eval_results.tsv`, `instances.log`, and
`instances.strip_term.log` (one eval row, five instance rows, five stripped
instance rows).

Compared with the verified serial main-result rerun
`20260524T160830 / 3upoqej5`, the batch path is lower on BLEU
(`33.3008 -> 30.0243`) and slightly lower on TERM_ACC (`0.6909 -> 0.6738`).
Treat this as a batch-diagnostic readout, not an automatic replacement for the
main table.
