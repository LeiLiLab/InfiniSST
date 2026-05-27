# Paper110 Extracted GS10K Glossary

## Hypothesis

For one-paper extracted-glossary evaluation, the runtime glossary should also have a large-bank control while keeping the metric denominator fixed to the paper extracted terms.

## Background / Motivation

The tagged ACL quick eval already supports fixed-denominator raw strict metrics for `raw` and `gs10k`.  Paper `2022.acl-long.110` needs the same noise-stress readout for the extracted glossary.

## What changed vs baseline

Build `expanded_glossary__2022.acl-long.110_gs10000.json` by preserving all extracted paper110 terms first and appending filler terms from the tagged ACL 10k bank until 10,000 entries.

## Expected metrics

No metrics are produced by this data-prep event.  Downstream eval should use the extracted paper110 glossary as the fixed metric denominator and compare runtime `extracted` vs `extracted_gs10k`.

## Verdict

Success.  Built 10,000 entries: 56 paper110 extracted terms preserved first, 9,944 filler terms appended, 15 duplicate filler terms skipped, and 0 missing-term entries skipped.
