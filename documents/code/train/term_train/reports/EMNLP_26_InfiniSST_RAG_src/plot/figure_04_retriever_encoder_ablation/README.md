# Figure 4: Retriever Encoder Ablation

- Paper label: `fig:ablation_retriever_encoder`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/retriever_encoder_ablation_devraw.pdf`
- Paper-local plotting script: `plot_figure_04_retriever_encoder_ablation.py`
- Frozen plotting data: `data.tsv`
- Local output snapshot: `retriever_encoder_ablation_devraw.pdf`,
  `retriever_encoder_ablation_devraw.png`

This package plots the retriever encoder ablation from the frozen 9-row
dev-raw TSV. The wrapper delegates plotting to the canonical retriever
analysis script with `--input-tsv data.tsv`, so it does not query WandB during
normal local regeneration.

Original provenance:

- Canonical plot script:
  `../../../../src/plot_retriever_encoder_ablation.py`
- Canonical data TSV:
  `../../../../reports/figures/20260524_retriever_encoder_ablation_devraw.tsv`
- Manifest:
  `../../../../manifests/2026/05/20260525T0153__analysis__retriever_encoder_ablation_figure4_style_refresh.json`
- Source W&B runs:
  `gczac4rf`, `deb5d6mn`, `q1eny6jt`

Regenerate the local snapshot from this directory:

```bash
python plot_figure_04_retriever_encoder_ablation.py
```

Regenerate and update the paper-facing figure:

```bash
python plot_figure_04_retriever_encoder_ablation.py --update-paper
```
