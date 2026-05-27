# Tagged ACL Clean V0 Same-LM Batch zh lm2 max80

## Hypothesis

For `zh/lm=2`, the cleaned tagged ACL glossary candidate should reduce noisy
daily-word terms while preserving the released tagged ACL terminology signal.
Using fixed `max_new_tokens=80` should avoid obvious `<term>`-tag truncation
without the larger behavioral shift of `max_new_tokens=256`.

## Background / Motivation

The raw tagged ACL glossary is released with ACL60/60, but it includes some
low-value daily-word entries.  This run tests
`acl6060_tagged_gt_raw_min_norm2_clean_candidate_v0.json` as both runtime
retrieval glossary and metric glossary under the same HN1024 tau=0.78 same-lm
batch runner used for the recent zh tagged ACL checks.

## What changed vs baseline

- Language: `zh`
- Latency multiplier: `lm=2`
- Runtime and metric glossary:
  `/mnt/gemini/data1/jiaxuanluo/glossary_clean_candidates/20260524_acl6060_tagged_raw/acl6060_tagged_gt_raw_min_norm2_clean_candidate_v0.json`
- Fixed `max_new_tokens=80`
- Same-lm batch runner with 5 ACL talks in one vLLM process
- HN1024 retriever at tau=0.78, top-k 10, look-back 1.92s

## Expected metrics

The main check is whether TERM_ACC improves or stays competitive after removing
noisy daily terms, and whether BLEU/StreamLAAL remain close to the raw-glossary
lm2 readout.  Because the metric denominator is now the cleaned candidate
glossary, this run is not directly denominator-equivalent to the raw main table.

## Verdict

Completed successfully as W&B `h3ced1i7`.

`zh/lm=2`, clean candidate v0, fixed `max_new_tokens=80`:

| BLEU | TERM_ACC | REAL_TERM_ADOPT | TERM_FCR | StreamLAAL |
| ---: | ---: | ---: | ---: | ---: |
| 48.11 | 89.26 | 92.08 | 11.11 | 1822.57 |

The cleaned glossary denominator has 540 evaluated term instances for this
setting, so this is a clean-glossary diagnostic rather than a denominator-matched
replacement for the released tagged-glossary main table.
