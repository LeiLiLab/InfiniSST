## Hypothesis

Quick readout for the V7/V8 Speech LLM refmatch SFT variants on tagged ACL zh lm=2 raw glossary.

## Background / Motivation

V7 uses refmatch-clean retriever term maps with target translations that exactly occur in the local/full reference. V8 uses the same data but serializes term maps as XML-style entries:

```text
<term>source => target</term>
```

This quick eval checks whether either variant improves TERM_ACC over the existing no-TM-SFT / LLM-generated SFT baselines before running a full sweep.

## What changed vs baseline

- Model: latest exported HF checkpoint under the V7 or V8 Aries training output roots.
- Eval setting: tagged ACL, `lang=zh`, `latency_multiplier=2`, `glossary=raw`.
- Retriever: `lh1b88kw` tau=0.73 timeline retrieval.
- Term map format:
  - V7: `plain`
  - V8: `xml_tagged`

## Expected metrics

Primary quick-check metric is TERM_ACC. Secondary metrics are BLEU, REAL_TERM_ADOPT, TERM_FCR, and StreamLAAL.

## Verdict

Completed on Aries GPUs 6,7 with run stamp `20260521T174131`.

| variant | W&B | BLEU | TERM_ACC | REAL_TERM_ADOPT | TERM_FCR | StreamLAAL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V7 refmatch plain | `mql4cqax` | 46.94 | 84.38 | 84.40 | 8.57 | 1679 |
| V8 refmatch XML | `ok1qp7dk` | 47.00 | 82.70 | 82.45 | 9.09 | 1729 |

V7 is the better quick-check candidate: it is substantially better than the earlier V3 retriever-SFT variants on TERM_ACC, while keeping TERM_FCR low. V8 XML tagging did not help in this setting; it slightly improves BLEU but hurts TERM_ACC and latency relative to V7.
