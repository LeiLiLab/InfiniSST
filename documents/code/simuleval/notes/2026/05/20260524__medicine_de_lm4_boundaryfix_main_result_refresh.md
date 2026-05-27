## Hypothesis

The medicine hardraw main-result figure should use the corrected boundaryfix
row for the `de`, `lm=4` InfiniSST/no-RAG baseline instead of the invalid
negative-latency batch output.

## Background / Motivation

The aries same-lm batch rerun completed, but the original German `instances.log`
concatenated generated chunks without boundary spaces. That made prediction word
counts diverge from delay counts and produced negative StreamLAAL. The run
manifest now records the invalid audit copies and the corrected boundaryfix TSV.

## What changed vs baseline

The canonical main-results builder now reads
`eval_results_streamlaal_term.hard_llm_manual_check.boundaryfix.tsv` for
`medicine_hardraw / InfiniSST / de / lm=4`. The old superseded serial launcher
for this row was removed, stale bytecode cache for the batch evaluator was
removed, the main-result TSV and figures were regenerated, and the paper results
text was updated to keep provisional/abnormal medicine rows visibly marked.

## Expected metrics

The canonical TSV should contain:

`BLEU=27.8196`, `StreamLAAL=2827.5546`, `StreamLAAL_CA=826.5604`,
`TERM_ACC=0.4735`, `TERM_CORRECT=340`, `TERM_TOTAL=718`.

## Verdict

Success. The canonical TSV was regenerated, `medicine_main_result.pdf/png` and
`new_main_result_tagged.pdf/png` were refreshed, and the paper rebuilt
successfully with `latexmk`. The invalid negative-latency TSV remains only as an
audit copy under `*.invalid_boundary_bug.*`; downstream table and figure
generation use the boundaryfix TSV.
