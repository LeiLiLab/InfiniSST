# Figure 7: Glossary-Bank Ablation

- Paper label: `fig:ablation_glossary_bank`
- LaTeX source: `latex/sections/results.tex`
- Figure output: `latex/figures/glossary_bank_ablation_zh_fixedraw.pdf`
- Inspection output: `latex/figures/glossary_bank_ablation_zh_fixedraw.png`
- Paper-local plotting script: `plot_figure_07_glossary_bank_ablation.py`
- Frozen plotting data: `data.tsv`
- Local output snapshot: `local/glossary_bank_ablation_zh_fixedraw.pdf`,
  `local/glossary_bank_ablation_zh_fixedraw.png`

This is the main-text Runtime Glossary-Bank ablation for En-Zh ACL tagged
evaluation. The runtime retrieval bank changes from raw to GS-1k and GS-10k,
while terminology metrics are scored against the fixed raw tagged denominator.
The figure plots only the `Tagged ACL` rows. The same TSV also contains
medicine rows that are used by the appendix table, not by Figure 7.

Original provenance:

- Collection/build script:
  `../../../../../../simuleval/src/build_glossary_bank_ablation_20260525.py`
- Canonical report TSV:
  `../../../../../../simuleval/reports/20260525_glossary_bank_ablation_zh_fixedraw_data.tsv`
- Manifest:
  `../../../../../../simuleval/manifests/2026/05/20260525T0122__analysis__glossary_bank_ablation_zh_fixedraw.json`

Regenerate from this directory:

```bash
python plot_figure_07_glossary_bank_ablation.py
```

The older top-level `glossary_bank_ablation_zh_fixedraw.pdf` and `.png`
snapshots may be owned by another local user on shared filesystems; the wrapper
therefore writes to `local/` by default.

A legacy `main_result_tagged.pdf` may also exist in this folder from an older
copy operation. It is not part of Figure 7; use
`../figure_01_main_result_tagged/` for the current Figure 1 package.

Regenerate and update the paper-facing figure:

```bash
python plot_figure_07_glossary_bank_ablation.py --update-paper
```
