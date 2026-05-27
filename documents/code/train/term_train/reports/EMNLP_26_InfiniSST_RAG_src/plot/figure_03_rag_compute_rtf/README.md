# Figure 3: RAG Compute RTF

- Paper label: `fig:rag_compute_rtf`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/rag_compute_rtf.pdf`
- Inspection output: `latex/figures/rag_compute_rtf.png`
- Paper-local plotting script: `plot_figure_03_rag_compute_rtf.py`
- Frozen plotting data: `data.tsv`
- Local output snapshot: `rag_compute_rtf.pdf`, `rag_compute_rtf.png`

This package tracks the RAG compute real-time-factor figure. The local
`data.tsv` is the frozen plotted summary. The wrapper delegates collection and
plotting to the canonical script, which re-reads the verified summary TSV and
runtime JSONL traces.

Original provenance:

- Canonical plot script:
  `../../../../src/plot_rag_compute_rtf.py`
- Canonical data TSV:
  `../../../../reports/figures/20260525_rag_compute_rtf.tsv`
- Runtime summary source:
  `../../../../../../simuleval/reports/20260525_glossary_bank_ablation_zh_fixedraw_data.tsv`
- Manifest:
  `../../../../manifests/2026/05/20260525T0149__analysis__rag_compute_rtf_figure.json`

Regenerate the local snapshot from this directory:

```bash
python plot_figure_03_rag_compute_rtf.py
```

Regenerate and update the paper-facing figure:

```bash
python plot_figure_03_rag_compute_rtf.py --update-paper
```
