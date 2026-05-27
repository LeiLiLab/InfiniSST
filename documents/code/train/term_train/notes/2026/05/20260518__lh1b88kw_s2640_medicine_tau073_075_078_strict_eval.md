# lh1b88kw s2640 medicine tau 0.73/0.75/0.78 strict readout

## Hypothesis

The dev-calibrated tau values can be reused without rerunning dev scoring. The only missing readout is strict MFA-only medicine recall and precision behavior at tau 0.73, 0.75, and 0.78.

## Background / Motivation

Medicine source labels were cleaned to drop fallback and non-MFA-matched terms. Prior tau sweeps on dev remain valid for tau selection, while the medicine readout must be refreshed against the strict MFA-only medicine dataset and glossary.

## What changed vs baseline

This is an eval-only medicine run from the `lh1b88kw` step-2640 checkpoint. It disables train-time negatives, leaves dev/ACL/tagged ACL JSONL paths empty, uses the strict MFA-only medicine JSONL and glossary, and sweeps only tau 0.73, 0.75, and 0.78.

## Expected metrics

Tau 0.0 strict medicine baseline is base 0.9522, gs1000 0.9489, and gs10000 0.9348 recall@10. This run should add filtered medicine readouts for the three reused dev-calibrated tau values without changing checkpoint or dataset definitions.

## Verdict

Completed in W&B run `qjy4m1x9` through taurus hold job `45269` step `45269.2`.
The run evaluated only strict MFA-only medicine (`dev=0`, `acl=0`,
`tagged_acl=0`, `medicine_dev=11071`) from the `lh1b88kw` step-2640
checkpoint.

Values are `Recall / P_micro / noise`.

| tau | base | gs1k | gs10k |
| --- | ---: | ---: | ---: |
| 0.73 | 92.57 / 13.49 / 2.34 | 92.52 / 13.06 / 2.57 | 92.15 / 11.41 / 3.73 |
| 0.75 | 91.11 / 14.93 / 1.83 | 91.11 / 14.53 / 1.95 | 90.86 / 12.75 / 2.74 |
| 0.78 | 89.00 / 18.43 / 1.14 | 89.00 / 18.13 / 1.18 | 89.00 / 16.21 / 1.53 |

The gs10000 bank reports `9994` active terms in this run because six expansion
terms were filtered by `eval_glossary_match_min_norm_chars=2`.
