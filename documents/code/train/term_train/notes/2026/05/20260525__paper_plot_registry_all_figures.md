## Hypothesis

Maintaining one paper-local figure registry will make figure edits less
ambiguous by linking every compiled paper figure to its LaTeX source, output
artifact, plotting script, data source, and provenance manifest.

## Background / Motivation

The paper currently mixes figures generated from simuleval analysis scripts,
retriever-analysis scripts, inline LaTeX placeholders, and static/copied assets.
Only Figure 7 had a paper-local plot package. Collaborators need a single list
before editing or regenerating figures.

## What changed vs baseline

Expanded `plot/README.md` into a complete registry for the current compiled
Figure 1--8 numbering from `acl_latex.aux`. The registry includes output files,
LaTeX labels, plotting scripts, data TSVs or static status, and manifest links.
It also records that Figure 6 is an inline LaTeX placeholder and that Figure 8
is currently a static copied artifact.

## Expected metrics

No experiment metrics change. Validation is path-level: the registered script
and data paths in `plot/README.md` should resolve from the paper plot directory.

## Verdict

Complete. The registry covers Figures 1--8 and the referenced reusable scripts
and stable data files resolve successfully.
