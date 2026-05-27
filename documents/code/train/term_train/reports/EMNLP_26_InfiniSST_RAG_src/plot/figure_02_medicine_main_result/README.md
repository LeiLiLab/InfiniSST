# Figure 2: Medicine Main Result

- Paper label: `fig:main_result_2`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/medicine_main_result.pdf`
- Inspection output: `latex/figures/medicine_main_result.png`
- Paper-local plotting script: `plot_figure_02_medicine_main_result.py`
- Frozen plotting data: `data.tsv`
- Local output snapshot: `medicine_main_result.pdf`, `medicine_main_result.png`

This package plots the medicine hardraw main-result panel from the frozen
`data.tsv` snapshot. The wrapper reuses the canonical paper plotting function
from `documents/code/simuleval/src/build_main_result_tables_and_figures_20260524.py`.

Original provenance:

- Canonical build script:
  `../../../../../../simuleval/src/build_main_result_tables_and_figures_20260524.py`
- Canonical data TSV:
  `../../../../../../simuleval/reports/20260524_main_result_data.tsv`
- Manifest:
  `../../../../../../simuleval/manifests/2026/05/20260524T1232__analysis__main_results_aclraw_medicine_figures.json`
- Later refresh manifests:
  `../../../../../../simuleval/manifests/2026/05/20260525T0120__analysis__offline_llm_upper_reference_main_results.json`,
  `../../../../../../simuleval/manifests/2026/05/20260525T140757__analysis__offline_acl_tagged_reference_recheck.json`,
  `../../../../../../simuleval/manifests/2026/05/20260525T2036__analysis__main_results_ja_medicine_cap16_update.json`

Regenerate the local snapshot from this directory:

```bash
python plot_figure_02_medicine_main_result.py
```

Regenerate and update the paper-facing figure:

```bash
python plot_figure_02_medicine_main_result.py --update-paper
```
