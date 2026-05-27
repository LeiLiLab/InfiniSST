# Tagged ACL New V10 sample50 HN1024 tau0.78 de lm2

## Hypothesis

The New V10 sample50 En-De lm2 BLEU drop at tau `0.79` may be caused by an
over-strict retrieval threshold. Re-running the same model and batch protocol at
tau `0.78` should test whether the dev-calibrated lower threshold restores BLEU
while keeping TERM_ACC above the no-RAG baseline.

## Background / Motivation

The completed tau `0.79` lm2 readout produced lower BLEU than the verified
InfiniSST/no-RAG and HN1024 reference rows. The same runtime retrieved slightly
fewer term references than tau `0.78` reference runs, so tau should be tested
before discarding the repaired no-GT term-map model.

## What changed vs baseline

Only the retrieval score threshold changes from `0.79` to `0.78`. The Speech
LLM, HN1024 retriever, tagged ACL raw glossary, five-talk same-LM batch shape,
max_new_tokens `80`, vLLM audio limit `128`, lookback `1.92s`, and strip-term
scoring remain fixed.

## Expected metrics

Gate condition: lm2 BLEU should recover toward the verified no-RAG lm2 baseline
(`30.0676`) while TERM_ACC remains clearly above no-RAG.

## Verdict

Completed as W&B `406eca6b`. The tau `0.78` retry did not recover BLEU:
BLEU `26.0605`, StreamLAAL `1438.9045`, StreamLAAL_CA `960.5404`, TERM_ACC
`0.8193` (`766/935`). Both `instances.log` and `instances.strip_term.log`
contain five rows.

Compared with tau `0.79` lm2, tau `0.78` increases the runtime reference density
back to the tau `0.78` reference-family level (`mean_refs=1.469`, nonempty rate
`0.672`, max `10`) but lowers BLEU further. This argues against tau `0.79`
being the primary cause of the New V10 lm2 BLEU drop.
