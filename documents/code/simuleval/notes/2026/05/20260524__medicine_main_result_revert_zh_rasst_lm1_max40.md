## Hypothesis

The zh medicine RASST lm=1 main-result point should use the earlier max40 batch result, because the max80 rerun degraded output quality and terminology accuracy.

## Background / Motivation

The max80 rerun completed, but its lm=1 readout was worse than the existing max40 batch result. The main-result figure and TSV therefore need to point back to the verified `20260524T0242` max40 eval artifact.

## What changed vs baseline

- Removed the max80 lm=1 override from `build_main_result_tables_and_figures_20260524.py`.
- Rebuilt `20260524_main_result_data.tsv`.
- Rebuilt `medicine_main_result.pdf`.
- The zh medicine RASST lm=1 row now points to the `20260524T0242` max40 eval result.

## Expected metrics

The `medicine_hardraw / RASST / zh / lm=1` row should report:

- BLEU: `33.4005`
- StreamLAAL: `1140.0656`
- TERM_ACC: `0.7905`
- TERM_CORRECT / TERM_TOTAL: `532 / 673`

## Verdict

SUCCESS. The main-result TSV and medicine figure were regenerated with the max40 zh medicine RASST lm=1 result.
