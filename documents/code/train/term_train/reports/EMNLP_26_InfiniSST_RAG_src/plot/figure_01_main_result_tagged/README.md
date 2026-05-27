# Figure 1: ACL Tagged Main Result

- Paper label: `fig:main_result_1`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/new_main_result_tagged.pdf`
- Inspection output: `latex/figures/new_main_result_tagged.png`
- Paper-local plotting script: `plot_figure_01_main_result_tagged.py`
- Frozen plotting data: `data.tsv`
- Local output snapshot: `new_main_result_tagged.pdf`, `new_main_result_tagged.png`

This package plots the ACL tagged raw-glossary main-result panel from the
frozen `data.tsv` snapshot. The wrapper reuses the canonical paper plotting
function from `documents/code/simuleval/src/build_main_result_tables_and_figures_20260524.py`.

Original provenance:

- Canonical build script:
  `../../../../../../simuleval/src/build_main_result_tables_and_figures_20260524.py`
- Canonical data TSV:
  `../../../../../../simuleval/reports/20260524_main_result_data.tsv`
- Manifest:
  `../../../../../../simuleval/manifests/2026/05/20260524T1232__analysis__main_results_aclraw_medicine_figures.json`
- Later refresh manifests:
  `../../../../../../simuleval/manifests/2026/05/20260525T0120__analysis__offline_llm_upper_reference_main_results.json`,
  `../../../../../../simuleval/manifests/2026/05/20260525T140757__analysis__offline_acl_tagged_reference_recheck.json`

Regenerate the local snapshot from this directory:

```bash
python plot_figure_01_main_result_tagged.py
```

Regenerate and update the paper-facing figure:

```bash
python plot_figure_01_main_result_tagged.py --update-paper
```
