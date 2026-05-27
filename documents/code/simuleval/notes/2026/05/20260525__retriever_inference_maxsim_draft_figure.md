## Hypothesis

A retriever-only inference figure will explain the current system better than
the old feature-pipeline diagram if it emphasizes timeline-aware look-back,
multi-scale MaxSim evidence, current-chunk overlap filtering, and step-local
term-map output.

## Background / Motivation

The current method text describes the online retriever as encoding the current
chunk plus a fixed `1.92s` look-back, computing MaxSim over multi-scale speech
windows, filtering evidence windows that do not overlap the current chunk, and
then applying threshold/top-K filtering. The existing rough diagram is too close
to an older speech-to-text feature similarity view and does not communicate
look-back or MaxSim clearly.

## What changed vs baseline

Create a paper-local editable draft package under
`plot/figure_00_retriever_inference_maxsim_draft/`. The preferred source is now
the clean SVG figure, with PDF/PNG previews exported from the SVG source. The
earlier HTML and PPTX drafts remain in the folder as legacy sketches. No LaTeX
source is modified yet.

## Expected metrics

Not applicable. This is a figure-drafting analysis event.

## Verdict

Generated a clean SVG version plus vector PDF and PNG previews under
`plot/figure_00_retriever_inference_maxsim_draft/`. The figure is intentionally
not inserted into LaTeX yet. Visual review passed for the updated design
objective: no gray panel frames, no step headings, visible look-back context,
semi-transparent candidate windows, MaxSim heatmap scoring, larger math labels,
compact filter condition, and concrete `Term_map` examples (`Shinzo Abe = 安倍晋三`
and `PM = 首相`). Playwright/Chromium exports the PDF directly from the SVG
source. The earlier HTML/PPTX drafts remain available but are no longer the
preferred editing surface.

Follow-up edit: tightened the layout, reduced arrowhead size, lengthened flow
arrows where possible, balanced the crowded right side by moving the MaxSim and
output blocks, made the candidate windows look like semi-transparent overlapping
virtual windows, enlarged `Filter`, and changed the output heading to
`TERM_MAP G_i`.
