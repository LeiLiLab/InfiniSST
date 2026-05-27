## Hypothesis

The Speech LLM ablation companion figures should use the same InfiniSST and
RASST points as the current tagged-ACL main-result snapshot.  The En-Ja
companion had stale InfiniSST/RASST rows after the main-result refresh, while
the En-Zh tau-comparison data should be regenerated from the current snapshot
to confirm it remains aligned.

## Background / Motivation

The paper-local Figure 6 package stores frozen TSV snapshots.  After the main
result was refreshed, these snapshots needed a consistency check against
`plot/figure_01_main_result_tagged/data.tsv`.

## What changed vs baseline

- Updated `data_ja.tsv` InfiniSST lm=1..4 rows from the current main-result
  tagged ACL baseline.
- Updated `data_ja.tsv` RASST lm=1..4 rows from the current main-result tagged
  ACL serial promptfix rows.
- Regenerated `data_zh_tau_compare.tsv` from the current main-result tagged ACL
  snapshot and the tau=0.0 comparison TSV.
- Regenerated and copied the En-Ja and En-Zh tau-comparison PDF/PNG figures to
  `latex/figures/`.

## Expected metrics

The En-Ja figure's InfiniSST and RASST rows should exactly match the current
`acl_tagged_raw` En-Ja rows in `figure_01_main_result_tagged/data.tsv`.  The
En-Zh tau=0.78 RASST and InfiniSST rows should also match the current En-Zh
main-result rows.

## Verdict

Completed.  Alignment validation passed for En-Ja InfiniSST/RASST and En-Zh
InfiniSST/RASST(tau=0.78).  `speechllm_ablation_ja.pdf` and
`speechllm_ablation_zh_tau_compare.pdf` were regenerated and copied to the
paper `latex/figures` directory.

Follow-up layout fix: the refreshed En-Ja InfiniSST lm=4 point has
StreamLAAL 3400.4659, which made the automatic x-axis show a 3500 ms tick and
visually compressed the plot.  The En-Ja figure was regenerated with
`--x-right-pad 25`, keeping the true lm=4 point visible while avoiding the
extra 3500 tick.
