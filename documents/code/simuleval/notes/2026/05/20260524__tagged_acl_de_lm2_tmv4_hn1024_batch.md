## Hypothesis

The previous term-map SFT En-De speech LLM may recover the older RASST BLEU
behavior when paired with the same HN1024 retriever, even if the newer clean de
MFA/source-filtered RASST model is lower on BLEU.

## Background / Motivation

The verified clean de RASST batch run improved TERM_ACC but was below InfiniSST
on BLEU. The older TM-SFT checkpoint is evaluated here as a direct batch
counterpart under the current HN1024 raw tagged ACL protocol.

## What changed vs baseline

This run uses
`/mnt/gemini/data/jiaxuanluo/owaski/gigaspeech-de-s_v4_ner_baseline_aligned_rate1.0_k20_final-bsz4`
with HN1024 retrieval. The protocol stays fixed: tagged ACL raw glossary, `de`,
`lm=2`, HN1024 checkpoint, `tau=0.78`, `top_k=10`, `lookback=1.92s`,
same-lm batch over five ACL talks, `max_new_tokens=80`, and assistant `<term>`
tag stripping before metrics.

## Expected metrics

The key comparison is BLEU and TERM_ACC against current clean de RASST batch
`lm=2`, origin+HN1024 `lm=2`, and the InfiniSST/no-RAG baseline `lm=2`.

## Verdict

Completed on Aries GPU 2,3 after retrying away from SYS-topology GPU pairs.

- BLEU: 31.1211
- StreamLAAL: 1618.2386
- StreamLAAL_CA: 1350.2295
- TERM_ACC: 0.8353 (781 / 935)

The output TSV and W&B run `40i1fl5v` are present. W&B emitted a local cache
warning for `/mnt/data7/... no space left on device`, but the metric logging
finished and the run has `status:success`.

Strip validation passed: raw and stripped logs both have 5 rows, source and
reference fields are unchanged, no `<term>` tags remain in the scored
`instances.strip_term.log`, and German word-level delay counts match the
stripped predictions.
