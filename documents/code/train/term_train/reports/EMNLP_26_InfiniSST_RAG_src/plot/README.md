# Paper Plot Registry

This directory is the paper-local management surface for figure assets. Each
current paper figure has a `figure_XX_*` folder with a README, frozen data or
static-output snapshot when available, and a local plotting entry point when a
reusable script exists.

The current figure numbers below come from the compiled
`acl_latex.aux`. If figure order changes, update this registry together with the
LaTeX edit.

## Registry

| Figure | Label | LaTeX source | Output | Paper-local plot folder | Plot script | Data source | Manifest / provenance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `fig:main_result_1` | `latex/sections/results.tex` | `latex/figures/new_main_result_tagged.pdf` and `.png` | `figure_01_main_result_tagged/` | `figure_01_main_result_tagged/plot_figure_01_main_result_tagged.py` | `figure_01_main_result_tagged/data.tsv`; canonical source `../../../../../simuleval/reports/20260524_main_result_data.tsv` | `../../../../../simuleval/manifests/2026/05/20260524T1232__analysis__main_results_aclraw_medicine_figures.json`; later refreshes include `20260525T0120__analysis__offline_llm_upper_reference_main_results.json`, `20260525T140757__analysis__offline_acl_tagged_reference_recheck.json`, `20260525T2110__analysis__acl_tagged_de_ja_infinisst_baseline_refresh.json` |
| 2 | `fig:main_result_2` | `latex/sections/results.tex` | `latex/figures/medicine_main_result.pdf` and `.png` | `figure_02_medicine_main_result/` | `figure_02_medicine_main_result/plot_figure_02_medicine_main_result.py` | `figure_02_medicine_main_result/data.tsv`; canonical source `../../../../../simuleval/reports/20260524_main_result_data.tsv` | `../../../../../simuleval/manifests/2026/05/20260524T1232__analysis__main_results_aclraw_medicine_figures.json`; later refreshes include `20260525T0120__analysis__offline_llm_upper_reference_main_results.json`, `20260525T140757__analysis__offline_acl_tagged_reference_recheck.json`, `20260525T2036__analysis__main_results_ja_medicine_cap16_update.json` |
| 3 | `fig:rag_compute_rtf` | `latex/sections/results.tex` | `latex/figures/rag_compute_rtf.pdf` and `.png` | `figure_03_rag_compute_rtf/` | `figure_03_rag_compute_rtf/plot_figure_03_rag_compute_rtf.py` | `figure_03_rag_compute_rtf/data.tsv`; canonical source `../../../reports/figures/20260525_rag_compute_rtf.tsv` | `../../../manifests/2026/05/20260525T0149__analysis__rag_compute_rtf_figure.json` |
| 4 | `fig:ablation_retriever_encoder` | `latex/sections/results.tex` | `latex/figures/retriever_encoder_ablation_devraw.pdf` | `figure_04_retriever_encoder_ablation/` | `figure_04_retriever_encoder_ablation/plot_figure_04_retriever_encoder_ablation.py` | `figure_04_retriever_encoder_ablation/data.tsv`; canonical source `../../../reports/figures/20260524_retriever_encoder_ablation_devraw.tsv` | `../../../manifests/2026/05/20260525T0153__analysis__retriever_encoder_ablation_figure4_style_refresh.json` |
| 5 | `fig:ablation_retriever_data` | `latex/sections/results.tex` | `latex/figures/retriever_data_ablation_dev.pdf` and `.png` | `figure_09_retriever_data_ablation/` | `figure_09_retriever_data_ablation/plot_figure_09_retriever_data_ablation.py` | `figure_09_retriever_data_ablation/data.tsv`; canonical source `../../../reports/figures/20260525_retriever_data_ablation_dev.tsv` | `../../../manifests/2026/05/20260525T2033__analysis__retriever_data_ablation_dev_figure.json` |
| 6 | `fig:ablation_hn_tau` | `latex/sections/results.tex` | `latex/figures/retriever_dev_pr_fixedraw_hn_comparison.pdf` | `figure_05_hn_tau_ablation/` | No reusable script preserved; see folder README before re-plotting. | `figure_05_hn_tau_ablation/data_multibank.tsv` and `figure_05_hn_tau_ablation/data_drop1_points.tsv` | `../../../manifests/2026/05/20260523T0430__analysis__hn_fixeddenom_report_denominator_prcurve_update.json`; caption refresh `../../../manifests/2026/05/20260525T0239__analysis__retriever_hn_tau_figure5_description_refresh.json` |
| 7 | `fig:ablation_speechllm` | `latex/sections/results.tex` | `latex/figures/speechllm_ablation_de.pdf` and `.png`; companion `latex/figures/speechllm_ablation_ja.pdf` and `.png`; tau companion `latex/figures/speechllm_ablation_zh_tau_compare.pdf` and `.png` | `figure_06_speechllm_placeholder/` | `figure_06_speechllm_placeholder/plot_figure_06_speechllm_ablation.py`; `figure_06_speechllm_placeholder/build_data_zh_tau_compare.py` | `figure_06_speechllm_placeholder/data.tsv`; `figure_06_speechllm_placeholder/data_ja.tsv`; `figure_06_speechllm_placeholder/data_zh_tau_compare.tsv`; legacy source `../../../../../simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst_data.tsv`; current source `figure_01_main_result_tagged/data.tsv`; tau=0.0 source `../../../../../simuleval/reports/20260526_tagged_acl_zh_tau000_vs_main_rasst.tsv` | `../../../../../simuleval/manifests/2026/05/20260521T0859__analysis__tagged_acl_main_results_llmgen_rasst_figure.json`; de refresh `../../../manifests/2026/05/20260525T1938__analysis__speechllm_ablation_de_figure6.json`; ja companion `../../../manifests/2026/05/20260525T1944__analysis__speechllm_ablation_ja_figure6_companion.json`; zh tau companion `../../../manifests/2026/05/20260526T0448__analysis__speechllm_ablation_zh_tau_compare.json` |
| 8 | `fig:ablation_glossary_bank` | `latex/sections/results.tex` | `latex/figures/glossary_bank_ablation_zh_fixedraw.pdf` and `.png` | `figure_07_glossary_bank_ablation/` | `figure_07_glossary_bank_ablation/plot_figure_07_glossary_bank_ablation.py` | `figure_07_glossary_bank_ablation/data.tsv`; canonical source `../../../../../simuleval/reports/20260525_glossary_bank_ablation_zh_fixedraw_data.tsv` | `../../../manifests/2026/05/20260525T055246__analysis__paper_plot_index_figure7.json`; canonical event `../../../../../simuleval/manifests/2026/05/20260525T0122__analysis__glossary_bank_ablation_zh_fixedraw.json` |
| 9 | `fig:ablation_multiscale_inference` | `latex/sections/results.tex` | `latex/figures/multiscale_inference_ablation_devraw.pdf` and `.png` | `figure_10_multiscale_inference_ablation/` | `figure_10_multiscale_inference_ablation/plot_figure_10_multiscale_inference_ablation.py` | `figure_10_multiscale_inference_ablation/data.tsv`; canonical source `../../../reports/figures/20260525_multiscale_inference_ablation_devraw.tsv` | `../../../manifests/2026/05/20260525T2100__analysis__multiscale_inference_ablation_report.json`; figure refresh `../../../manifests/2026/05/20260525T2130__analysis__multiscale_inference_ablation_paper_figure.json` |
| 10 | `fig:term_duration` | `latex/sections/appendix.tex` | `latex/figures/term_duration_dist.pdf` | `figure_08_term_duration_distribution/` | Data-prep script snapshot only: `figure_08_term_duration_distribution/compute_term_duration_distribution.py`; plotting script not recovered. | Static PDF snapshot; no local data TSV/JSON recovered yet. | Not yet backfilled for this paper figure |

## Non-Figure Visuals

The introduction example is `tab:intro`, not a figure. The context-duration
ablation is `tab:ablation_context`, not a figure. They are maintained directly
inside `latex/sections/*.tex`.

Unassigned draft: `figure_00_retriever_inference_maxsim_draft/` contains a
clean SVG retriever inference diagram focused on look-back, semi-transparent
candidate windows, MaxSim scoring, compact filtering, and concrete `Term_map`
examples. The preferred source is `retriever_inference_maxsim_clean.svg`, with
vector PDF output generated by `draw_retriever_inference_maxsim_svg.py`. It is
not inserted into LaTeX yet. Provenance:
`../../../../../simuleval/manifests/2026/05/20260525T163317__analysis__retriever_inference_maxsim_draft_figure.json`.

## Maintenance Rules

For every new or refreshed paper figure:

1. Add or update one row in the registry above.
2. Prefer a paper-local folder named `figure_XX_short_name/`.
3. Put quick-edit plotting code, frozen plotting data, and output snapshots in
   that folder.
4. Keep the original experiment aggregation script in its module directory and
   link it from the per-figure README for provenance.
5. Register the change with an analysis manifest when the figure or data changes.
