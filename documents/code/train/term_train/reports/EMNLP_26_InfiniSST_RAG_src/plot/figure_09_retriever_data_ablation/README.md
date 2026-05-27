# Figure 9: Retriever Data Ablation

- Paper label: `fig:ablation_retriever_data`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/retriever_data_ablation_dev.pdf` and `.png`
- Paper-local plotting script: `plot_figure_09_retriever_data_ablation.py`
- Frozen plotting data: `data.tsv`
- Canonical data snapshot: `../../../reports/figures/20260525_retriever_data_ablation_dev.tsv`

This figure compares the main retriever trained with GigaSpeech-derived pairs
plus Wiki-synthetic terminology supervision against a GigaSpeech-only
ablation. The metric is the main retriever development readout, not ACL.

Metrics are from `wandb_tool.py db-compare --refresh --anchor-metric both`:

- Main run: `lh1b88kw`, primary bundle at step 1440.
- GigaSpeech-only run: `g49qabuf`, primary bundle at step 240.

For the GigaSpeech-only run, W&B logs the thresholded values under the
`topk10_chunk_any_positive_filtered_recall` key. The TSV normalizes that to the
same plotted filtered-recall columns used for the main run.

Regenerate the local snapshot from this directory:

```bash
python plot_figure_09_retriever_data_ablation.py
```

Regenerate and update the paper-facing figure:

```bash
python plot_figure_09_retriever_data_ablation.py --update-paper
```
