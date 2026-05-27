## Hypothesis

The low BLEU in the Japanese exact GT-term-wrapped TM-SFT readout for `lm=1`
and `lm=4` is caused by streaming repetition under the historical
`term_map: NONE` empty-retrieval prompt.  Omitting the term-map block when
retrieval returns no references should better match the TM-SFT training data
and reduce repeated continuations.

## Background / Motivation

The first Japanese tagged-ACL HN1024 readout completed for `lm=1..4`, but
`lm=1` and `lm=4` had anomalously low BLEU.  Inspection of
`instances.strip_term.log` showed severe over-generation: `lm=1` over-generated
multiple talks, and `lm=4` entered an "online prefix" repetition loop for
`2022.acl-long.110`.

## What changed vs baseline

Only the empty-retrieval prompt policy changes:

- previous: `empty_term_map_policy=none_block`, rendering `term_map: NONE`
- this diagnostic: `empty_term_map_policy=omit`, leaving empty-retrieval user
  turns audio-only

All other settings stay fixed: same Japanese exact GT-term-wrapped TM-SFT
model, HN1024 retriever, tagged ACL raw glossary, `tau=0.78`,
`max_new_tokens=80`, `VLLM_LIMIT_AUDIO=128`, and `VLLM_MAX_MODEL_LEN=12288`.

## Expected metrics

The immediate target is not a new paper-facing row, but a controlled diagnosis.
If the abnormal length ratios and repetition disappear, rerun the complete
`lm=1..4` curve under one consistent prompt policy before using the result.

## Verdict

Pending.
