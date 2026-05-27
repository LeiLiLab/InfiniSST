## Hypothesis

The origin no-TM-SFT En-De speech LLM with the HN1024 retriever may preserve
more BLEU than the clean de RASST SFT model while still gaining term accuracy
from retrieval.

## Background / Motivation

The current verified clean de RASST tagged-ACL raw readout improves TERM_ACC but
falls behind InfiniSST/no-RAG in BLEU. This probe isolates whether the drop comes
from term-map SFT itself by running the origin En-De SLM with the same HN1024
retriever and tagged ACL raw glossary.

## What changed vs baseline

This run uses `/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_origin-bsz4`
with RAG enabled. The protocol stays fixed: tagged ACL raw glossary, `de`,
`lm=2`, HN1024 checkpoint, `tau=0.78`, `top_k=10`, `lookback=1.92s`,
same-lm batch over five ACL talks, `max_new_tokens=80`, and assistant `<term>`
tag stripping before metrics.

## Expected metrics

The key comparison is BLEU and TERM_ACC against current clean de RASST batch
`lm=2` and the InfiniSST/no-RAG baseline `lm=2`.

## Verdict

Completed on Aries GPU 0,1. The eval output has 5 `instances.log` rows and 5
`instances.strip_term.log` rows. Metrics from `eval_results.tsv`:

- BLEU: 30.8132
- StreamLAAL: 1604.7783
- StreamLAAL_CA: 1299.0939
- TERM_ACC: 0.8225 (769 / 935)

W&B run: `simuleval_eval/p21ne026`. The wrapper stderr reported a local W&B
cache logger no-space warning under `/mnt/data7`, but W&B still synced and the
eval TSV was written successfully.
