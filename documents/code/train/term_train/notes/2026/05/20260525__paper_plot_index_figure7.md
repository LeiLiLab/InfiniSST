## Hypothesis

A paper-local plot directory with frozen data and a lightweight plotting entry
point makes Figure 7 easier to edit without re-running remote aggregation.

## Background / Motivation

Figure 7 currently corresponds to `fig:ablation_glossary_bank` and the output
`latex/figures/glossary_bank_ablation_zh_fixedraw.pdf`. The original
aggregation script lives under `documents/code/simuleval/src/` and also handles
data collection and appendix table generation. For paper editing, collaborators
need a smaller entry point next to the paper source.

## What changed vs baseline

Added `plot/` under the EMNLP paper source directory. The Figure 7 subdirectory
contains a README, a frozen TSV copied from the canonical simuleval report, and
a paper-local matplotlib script that regenerates the Figure 7 PDF/PNG from that
TSV.

## Expected metrics

No experiment metric changes. The expected validation is that the paper-local
script regenerates `latex/figures/glossary_bank_ablation_zh_fixedraw.pdf` and
the matching PNG without requiring SSH or remote collection.

## Verdict

Complete. The paper-local Figure 7 script ran successfully and rewrote the PDF
and PNG from the frozen TSV snapshot.
