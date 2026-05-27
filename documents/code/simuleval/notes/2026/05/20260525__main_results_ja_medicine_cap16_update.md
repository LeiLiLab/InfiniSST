# Analysis: Main Results JA Medicine Cap16 Update

## Hypothesis
The completed JA cap16-denoise medicine hardraw batch should replace the previous JA RASST medicine placeholders in the main-result TSVs and paper Figure 2.

## Background / Motivation
The medicine main-result figure previously marked En-Ja RASST rows as unavailable. The JA medicine batch eval `20260525T1840__simuleval__medicine_ja_cap16_denoise_lm1234_batch_taurus` completed with verified per-LM `eval_results.tsv`, `instances.log`, and `instances.strip_term.log` artifacts.

## What changed vs baseline
- Replaced the four `medicine_hardraw / RASST / ja / lm=1..4` placeholder rows in the canonical main-result TSV with verified eval rows.
- Synchronized the paper-local Figure 1 and Figure 2 data snapshots with the canonical TSV.
- Updated the canonical builder so medicine RASST De/Ja rows rebuild from the current cap16-denoise verified artifacts instead of stale dirty/provisional or placeholder paths.
- Regenerated `medicine_main_result.pdf` and `.png`, both local to the plot package and under `latex/figures/`.
- Removed outdated red text that said complete En-Ja medicine rows were unavailable.

## Expected metrics
The updated JA medicine RASST rows should report:

| lm | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | TERM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 18.4229 | 1474.4801 | 1650.9138 | 0.7577 | 491/648 |
| 2 | 24.6735 | 2186.5425 | 1767.1576 | 0.8102 | 525/648 |
| 3 | 26.8288 | 2781.0621 | 1782.7240 | 0.8380 | 543/648 |
| 4 | 28.6781 | 3199.2583 | 1708.7238 | 0.8349 | 541/648 |

## Verdict
Success. The canonical TSV, paper-local TSV snapshots, builder script, results text, and paper-facing `medicine_main_result.pdf/png` were updated from verified JA medicine eval artifacts. A temporary full rebuild check passed and reproduced the current verified De/Ja medicine RASST rows.

