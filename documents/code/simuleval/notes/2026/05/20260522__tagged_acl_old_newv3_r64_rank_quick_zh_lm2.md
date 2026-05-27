# Tagged ACL quick rank readout for old new_v3 r64/a128

## Hypothesis

Evaluating old `new_v3` r64/a128 checkpoints against the same tagged ACL fast
settings as r32/a64 isolates whether extra LoRA capacity improved term adoption
or mainly changed BLEU/latency.

## Background / Motivation

Existing quick readouts cover old `new_v3` r32/a64.  The r64/a128 checkpoints
`q159wce4` and `rj1v1p7r` were only available as MCore checkpoints, so they are
first exported to HF and then evaluated with vLLM.

## What changed vs baseline

For each model, this launcher runs:

- full-corpus tagged ACL `zh lm=2 raw`
- one-paper `2022.acl-long.110` with the extracted per-paper glossary as the
  retrieval glossary

The fixed metric denominator for tagged strict terms remains the raw tagged ACL
glossary unless the downstream eval script is explicitly changed.

## Expected metrics

Compare against old r32/a64 quick rows:

- full new_v3 r32/a64: W&B `538mqu47`
- random new_v3 r32/a64: W&B `u3v3lii2`

## Verdict

Pending eval.
