## Hypothesis

The low `de/lm=1` tagged-ACL RASST row is partly caused by an empty-term-map
prompt mismatch. Training represents empty chunks as bare `<audio>` messages,
whereas the current batched eval emits explicit `term_map:\nNONE`. Omitting the
empty term-map block may reduce the `2022.acl-long.367` decode loop and recover
BLEU.

## Background / Motivation

The previous `de/lm=1` readout with TM-SFT exact GT-term wrapping and HN1024
tau 0.78 produced BLEU 20.6877 and TERM_ACC 0.7872. Diagnosis showed that the
drop is dominated by over-generation on `2022.acl-long.367`, with many
empty-retrieval calls and no Chinese pollution in retrieved references.

## What changed vs baseline

This run changes only the empty retrieval prompt policy:

- baseline policy: emit explicit `term_map:\nNONE`;
- test policy: omit the term-map text block when retrieval returns no usable
  references, leaving the user message audio-only.

All other settings remain fixed: German tagged ACL raw glossary, TM-SFT exact
GT-term-wrapped Speech LLM, HN1024 retriever, tau 0.78, top-k 10, lookback 1.92s,
`max_new_tokens=80`, same-lm batch eval, and `lm=1` only.

## Expected metrics

If the empty-term-map mismatch is a material trigger, `lm=1` BLEU should improve
and the repeated tail on `2022.acl-long.367` should weaken or disappear. TERM_ACC
may move but should remain clearly above no-RAG.

## Verdict

Completed. Omitting empty term-map blocks improves `de/lm=1` from BLEU 20.6877
to 24.0580 and TERM_ACC from 0.7872 to 0.8171 (764/935), but it does not recover
the lm2/lm3 BLEU range.

The original pathological talk `2022.acl-long.367` is repaired: prediction/ref
length ratio drops from 2.04 to 1.04, repeated 8-gram rate drops from 0.539 to
0.000, and the mixed `位置编码` / `Rei儿Bemerkungen` tail disappears. Thus
explicit `term_map:\nNONE` is a material trigger for the lm1 decode loop, but
additional low-latency degradation remains.
