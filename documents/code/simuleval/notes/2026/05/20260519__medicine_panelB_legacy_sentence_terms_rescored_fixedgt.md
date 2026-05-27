## Hypothesis

Legacy medicine oracle outputs can still be used as Panel B / appendix evidence
if their metrics are recomputed against a glossary-derived source of truth.

## Background / Motivation

The old oracle term_map prompts were derived from ESO `sentences[*].terms`, which
is incomplete and should not define TERM_ACC or false-copy denominators.
For reporting, the metric glossary must be derived from translated medicine GT
glossary entries matched against source and reference sentences.

## What changed vs baseline

- Baseline output: existing `medicine_onetalk_oracle_gt_sft_oraclegt_r32a64_20260519`
  and `medicine_remaining_oracle_gt_sft_oraclegt_r32a64_20260519` generated
  hypotheses.
- Diff: reused legacy `instances.log` and runtime logs, but regenerated
  per-sample metric glossaries with `term_source=glossary_match` and
  `glossary_source_filter=medicine_gt`.
- Important limitation: prompts remain legacy `sentence_terms` oracle prompts;
  only the metric denominator and adoption/FCR computation are corrected.

## Expected metrics

lm1-lm3 should have complete four-sample summaries.  lm4 is expected to be
partial because legacy sample `545006` did not finish.

## Verdict

Rescore completed.  lm1-lm3 have complete four-sample summaries; lm4 is marked
partial with missing sample `545006`.
