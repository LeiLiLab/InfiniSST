# Figure 5: HN / Tau Ablation

- Paper label: `fig:ablation_hn_tau`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/retriever_dev_pr_fixedraw_hn_comparison.pdf`
- Paper-local plotting script: none preserved
- Frozen plotting data: `data_multibank.tsv`, `data_drop1_points.tsv`
- Local output snapshot: `retriever_dev_pr_fixedraw_hn_comparison.pdf`,
  `retriever_dev_pr_fixedraw_hn_comparison.png`

This package collects the stable TSV inputs and output snapshot for the
retriever hard-negative and tau-calibration figure. The original reusable
plotting script was not preserved; the current paper PDF matches the canonical
report PDF snapshot.

Original provenance:

- Stable multibank TSV:
  `../../../../reports/figures/20260523_dev_multibank_pr_fixedraw_commonbase_nohn_hn256_hn1024.tsv`
- Stable drop-point TSV:
  `../../../../reports/figures/20260523_dev_pr_fixedraw_commonbase_hn1024_drop1_points.tsv`
- Canonical report PDF:
  `../../../../reports/figures/20260523_dev_pr_fixedraw_commonbase_threepanel_nohn_hn256_hn1024.pdf`
- Manifest:
  `../../../../manifests/2026/05/20260523T0430__analysis__hn_fixeddenom_report_denominator_prcurve_update.json`
- Caption refresh manifest:
  `../../../../manifests/2026/05/20260525T0239__analysis__retriever_hn_tau_figure5_description_refresh.json`

If this figure needs editing, start from the two frozen TSVs here and create a
new local plotting script in this folder.
