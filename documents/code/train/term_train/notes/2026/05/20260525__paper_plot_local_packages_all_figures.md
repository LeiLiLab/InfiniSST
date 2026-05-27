## Hypothesis

Moving each current paper figure into a paper-local `figure_XX_*` package will
make plot maintenance less ambiguous than using a top-level registry that only
points to scattered scripts, TSVs, and static PDFs.

## Background / Motivation

The paper-local `plot/` directory already had a registry for Figures 1--8, but
only Figure 7 had a full local folder with README, data, script, and output
snapshots. The remaining figures were discoverable only through registry rows,
canonical report paths, or static PDF references.

## What changed vs baseline

Created per-figure folders for the current compiled figure numbering:

- `figure_01_main_result_tagged/`
- `figure_02_medicine_main_result/`
- `figure_03_rag_compute_rtf/`
- `figure_04_retriever_encoder_ablation/`
- `figure_05_hn_tau_ablation/`
- `figure_06_speechllm_placeholder/`
- `figure_08_term_duration_distribution/`

Updated `plot/README.md` so every Figure 1--8 row points first to its local
package. Figures 1--4 now have local wrapper scripts, frozen TSV snapshots, and
local output snapshots. Figure 7 keeps its existing package and now supports
`--update-paper`. Figure 5 records stable TSVs plus the static output snapshot
because the reusable plotting script was not preserved. Figure 6 remains an
inline placeholder. Figure 8 records the static PDF and data-prep script
snapshot, with the missing plotting script/data called out explicitly.

## Expected metrics

No experiment metrics change. Validation is path-level and script-level:
package files should exist, registry paths should resolve, and local wrapper
scripts for Figures 1--4 and 7 should compile and regenerate local snapshots.

## Verdict

Complete. The paper-local `plot/` directory now has one management folder per
current paper figure, with provenance and limitations recorded in each folder
README and the top-level registry. Local regeneration was run for Figures 1--4
and 7 without updating the paper-facing `latex/figures/*` files.
