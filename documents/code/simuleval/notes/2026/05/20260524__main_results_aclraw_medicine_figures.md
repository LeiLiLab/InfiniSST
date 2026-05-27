## Hypothesis

The paper should report main results from a single canonical TSV that separates
ACL tagged raw, ACL paper-extracted, and medicine hardraw rows. New paper plots
should use the fixed ACL tagged raw denominator and the medicine hardraw
denominator, rather than reusing the older placeholder figures.

## Background / Motivation

The previous paper figures mixed older placeholder results with newer HN1024
tau=0.78 readouts. ACL tagged raw RASST now has serial verified artifacts for
zh, de, and ja. Medicine hardraw has clean zh RASST rows, reusable hard-manual
InfiniSST baseline rows where the five-sample check passes, dirty de RASST rows
that can be shown only as provisional, and missing offline / ja RASST rows that
must remain explicit placeholders.

## What changed vs baseline

- Added `documents/code/simuleval/reports/20260524_main_result_data.tsv` as the
  canonical main-result table.
- Added `documents/code/simuleval/src/build_main_result_tables_and_figures_20260524.py`
  to validate the table and regenerate figures from it.
- Generated `new_main_result_tagged` for ACL tagged raw and `medicine_main_result`
  for medicine hardraw.
- Marked temporarily unavailable or abnormal medicine rows in red font in the
  figure and corresponding paper caveat.
- Updated the paper to reference the new figure names while leaving the older
  `main_result_tagged.pdf` and `main_result_paper.pdf` files untouched.

## Expected metrics

The ACL tagged raw RASST rows should come from verified `eval_results.tsv`
artifacts, including the zh lm1 max256 same-lm batch readout from
20260524T1442 / W&B `kolja8vr` and zh lm2-4 rows from the 20260524T0522 and
20260524T0555 runs. Medicine hardraw should plot clean zh RASST rows, validated
InfiniSST baseline rows, and dirty/provisional de RASST rows with an explicit
caveat. Missing rows should be `NA` with placeholder status, not silently
filled.

## Verdict

Completed. The canonical TSV contains ACL tagged raw, ACL paper-extracted, and
medicine hardraw rows with unique `(dataset, method, lang, lm)` keys. The new
ACL tagged raw figure and medicine hardraw figure were generated as PDF and PNG.
ACL tagged raw RASST zh lm1 was updated to the user-requested max256 same-lm
batch readout (`kolja8vr`): BLEU 44.41, TERM_ACC 85.39, StreamLAAL 1236.72.
Medicine de RASST rows are marked `dirty_untrusted`; medicine offline, missing
baseline rows, and medicine ja RASST remain explicit `NA` placeholders. These
temporarily unavailable or abnormal medicine rows are visually called out in red
font in the figure and paper text. A follow-up provenance check found that
medicine InfiniSST de/lm4 was not missing from the plot by accident: its Taurus
generation attempt failed during vLLM EngineCore initialization before
`instances.log` or a hard-manual `eval_results_streamlaal_term` TSV was written,
so it remains an explicit failed-baseline placeholder.
