## Hypothesis

Use the latest cap16-denoise de RASST readouts for the paper-facing ACL tagged and medicine hardraw main-result figures.

## Background / Motivation

The previous de RASST rows used older NewV9/MFA or dirty medicine outputs. New cap16-denoise tagged-term SLM runs are available for tagged ACL and medicine hardraw, with verified `eval_results.tsv` files.

## What changed vs baseline

- Replaced `acl_tagged_raw / RASST / de / lm=1..4` with cap16-denoise outputs.
- Replaced `medicine_hardraw / RASST / de / lm=1..4` with cap16-denoise medicine hardraw outputs.
- Synchronized the canonical TSV into the two paper-local figure data snapshots.
- Regenerated `new_main_result_tagged` and `medicine_main_result` PDF/PNG figures.

## Expected metrics

Tagged ACL de RASST should use TERM_TOTAL=935. Medicine hardraw de RASST should use TERM_TOTAL=647.

## Verdict

Success. TSV uniqueness and numeric validation passed for 87 rows. The eight updated de RASST rows match their verified `eval_results.tsv` values and both paper figures were regenerated.
