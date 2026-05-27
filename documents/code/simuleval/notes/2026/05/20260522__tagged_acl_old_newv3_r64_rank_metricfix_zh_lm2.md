# Tagged ACL old new_v3 r64/a128 metric-glossary fix

## Hypothesis

The previous `paper110_extracted` quick readout used the extracted glossary for
retrieval but kept the raw tagged ACL glossary as the metric denominator.  This
mixed denominator includes strict terms such as `utterance` that are absent from
the extracted retrieval glossary, so the extracted-glossary result is not the
intended reference.

## Background / Motivation

For the one-paper extracted-glossary reference, retrieval glossary and metric
glossary should match unless the setting is explicitly labelled as a fixed
tagged-denominator stress test.

## What changed vs baseline

This rerun sets `EVAL_GLOSSARY_FOLLOWS_KIND_OVERRIDE=1` in the tagged ACL
launcher.  Therefore:

- `raw` still uses the fixed raw tagged glossary for retrieval and metrics.
- `extracted` uses `extracted_glossary__2022.acl-long.110.json` for both
  retrieval and TERM/REAL/FCR metric denominators.

The q159 full raw result from the previous run remains valid and is not rerun.
This launcher reruns q159 paper110 extracted and the missing rj1 full/raw plus
rj1 paper110 extracted rows.

## Expected metrics

The corrected extracted one-paper TERM_ACC should no longer be penalized for
terms that are absent from the extracted glossary denominator.

## Verdict

Pending eval.
