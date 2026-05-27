## Hypothesis

The ACL tagged raw offline horizontal references should be rebuilt from
`/mnt/data/siqiouyang/runs/infinisst_rag/offline` and rescored against the
raw tagged ACL glossary, because the earlier dashed-line BLEU and TERM_ACC
values were suspected to be stale or mismatched.

## Background / Motivation

Figure 1 (`fig:main_result_1`) plots two offline full-context reference lines
for ACL tagged raw: `Offline ST` and `Offline + GT terms`. These lines are
derived from Siqi's offline outputs and should be validated from
`scores.tsv` plus `instances.log`, not copied from prior notes.

## What changed vs baseline

This is a paper-figure refresh/recheck only. It reruns
`documents/code/simuleval/src/build_main_result_tables_and_figures_20260524.py`
using the current files under `/mnt/data/siqiouyang/runs/infinisst_rag/offline`.
No streaming model output or glossary file is changed.

## Expected metrics

Expected ACL tagged raw offline references after recomputation:

| lang | method | BLEU | TERM_ACC | TERM_CORRECT | TERM_TOTAL |
| --- | --- | ---: | ---: | ---: | ---: |
| zh | Offline ST | 51.3160 | 0.7551 | 672 | 890 |
| zh | Offline + GT terms | 54.6110 | 0.9438 | 840 | 890 |
| de | Offline ST | 38.6910 | 0.7230 | 676 | 935 |
| de | Offline + GT terms | 41.4850 | 0.9080 | 849 | 935 |
| ja | Offline ST | 35.0730 | 0.6830 | 642 | 940 |
| ja | Offline + GT terms | 38.6200 | 0.9521 | 895 | 940 |

## Verdict

Regenerated the shared main-result TSV, refreshed the paper figure PDFs/PNGs,
and rebuilt `acl_latex.pdf` successfully.

The recomputed ACL tagged raw dashed-line values match the values already
encoded in the current TSV:

| lang | method | BLEU | TERM_ACC | TERM_CORRECT | TERM_TOTAL |
| --- | --- | ---: | ---: | ---: | ---: |
| zh | Offline ST | 51.3160 | 0.7551 | 672 | 890 |
| zh | Offline + GT terms | 54.6110 | 0.9438 | 840 | 890 |
| de | Offline ST | 38.6910 | 0.7230 | 676 | 935 |
| de | Offline + GT terms | 41.4850 | 0.9080 | 849 | 935 |
| ja | Offline ST | 35.0730 | 0.6830 | 642 | 940 |
| ja | Offline + GT terms | 38.6200 | 0.9521 | 895 | 940 |

The same generator also refreshed `medicine_main_result.pdf` and
`medicine_main_result.png`. The recomputed medicine hardraw offline references
are:

| lang | method | BLEU | TERM_ACC | TERM_CORRECT | TERM_TOTAL |
| --- | --- | ---: | ---: | ---: | ---: |
| zh | Offline ST | 45.5100 | 0.4301 | 320 | 744 |
| zh | Offline + GT terms | 47.5850 | 0.9059 | 674 | 744 |
| de | Offline ST | 34.3600 | 0.4889 | 351 | 718 |
| de | Offline + GT terms | 35.8160 | 0.8621 | 619 | 718 |
| ja | Offline ST | 32.1240 | 0.3220 | 237 | 736 |
| ja | Offline + GT terms | 34.0710 | 0.8519 | 627 | 736 |
