# Glossary Bank Ablation, zh, Fixed Raw Denominator

## Hypothesis

Expanding the runtime retrieval glossary from the raw task glossary to GS-1k and
GS-10k should reveal whether the final HN1024 + New V9 RASST system remains
stable under additional distractor terms when the metric denominator is fixed.

## Background / Motivation

The paper needs a reviewer-defensible glossary-bank ablation for En-Zh on both
tagged ACL and the strict medicine readout.  The ablation must not change the
TERM metric denominator when the runtime bank changes.  Tagged ACL already had a
complete raw/GS-1k/GS-10k merged report using strip-term recheck.  Medicine GS
rows were generated on PSC, but their staged reference text differed from the
medicine raw main-result reference; therefore their original PSC `eval_results`
had `TERM_TOTAL=739` instead of the raw main-result denominator `673`.

## What changed vs baseline

- Added a local re-posteval script that fetches PSC medicine GS
  `instances.log` and runtime JSONL files, then re-scores them against the same
  local raw source, reference, and glossary files used by the medicine raw main
  result.
- Added a glossary-bank ablation builder that merges tagged ACL raw/GS rows,
  medicine raw rows, and medicine GS re-posteval rows.
- Added a paper figure at
  `latex/figures/glossary_bank_ablation_zh_fixedraw.pdf`.
- Updated `results.tex` to replace the online-SimulEval placeholder with the
  glossary-bank ablation figure and text.

## Expected metrics

The final complete table should contain 24 rows:

- datasets: `Tagged ACL`, `Medicine`
- runtime banks: `raw`, `gs1k`, `gs10k`
- latency multipliers: `lm=1,2,3,4`

Each row records BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, TERM_FCR, and the
source TSV path.  All medicine rows should have `TERM_TOTAL=673`; all tagged ACL
rows use the tagged raw fixed denominator from the merged tagged report.

## Verdict

Complete as of 2026-05-25 02:26 UTC: 24/24 rows are collected under the fixed
raw-denominator policy.  The paper-facing figure is intentionally narrowed to
the En-Zh ACL tagged rows so the main ablation remains compact.  The full
ACL-plus-medicine table is generated at
`latex/tables/glossary_bank_ablation_zh_fixedraw_appendix.tex` and included in
Appendix `Runtime Glossary-Bank Ablation Details`.

The medicine GS rows are kept as a stricter domain stress test rather than the
main paper figure.  The `gs10k lm4` low TERM_ACC investigation found that the
drop is not a denominator/post-eval bug; the larger bank introduces
near-duplicate and variant translations that can reduce strict exact raw-target
matches even when BLEU remains stable.
