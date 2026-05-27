# Multi-Scale Inference Ablation Figure

This folder contains the paper-local snapshot for the multi-scale inference
ablation figure.

- `data.tsv` is frozen from
  `../../../reports/figures/20260525_multiscale_inference_ablation_devraw.tsv`.
- `plot_figure_10_multiscale_inference_ablation.py` regenerates local PDF/PNG
  outputs and can copy them into `latex/figures/` with `--update-paper`.
- The main plot uses the three primary rows: multi-scale MaxSim, largest-only
  MaxSim-window inference, and the historical dense 1.92s single-embedding
  trained retriever. The auxiliary fixed-5.76s checkpoint diagnostic is kept in
  the TSV for provenance but excluded from the plotted bars.

Primary provenance:

- `../../../manifests/2026/05/20260525T2100__analysis__multiscale_inference_ablation_report.json`
- `../../../manifests/2026/05/20260525T013600__retriever_eval__context_ablation_varctx_perctxraw_100k.json`
- `../../../manifests/2026/05/20260525T204630__retriever_eval__maxsim_w24_lh1b88kw_varctx_devraw_100k.json`
- `../../../manifests/2026/05/20260525T205627__retriever_eval__dense_ctx192_r5l4780c_devraw_100k.json`
