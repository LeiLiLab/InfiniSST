# Figure 6: Speech LLM Ablation

- Paper label: `fig:ablation_speechllm`
- LaTeX source: `latex/sections/results.tex`
- Paper output: `latex/figures/speechllm_ablation_de.pdf`
- Additional En-Ja output: `latex/figures/speechllm_ablation_ja.pdf`
- Inspection outputs: `latex/figures/speechllm_ablation_de.png`,
  `latex/figures/speechllm_ablation_ja.png`
- Paper-local plotting script: `plot_figure_06_speechllm_ablation.py`
- En-Zh tau-comparison data builder: `build_data_zh_tau_compare.py`
- Frozen plotting data: `data.tsv` (En-De), `data_ja.tsv` (En-Ja),
  `data_zh_tau_compare.tsv` (En-Zh five-line tau comparison)
- Local output snapshot: `speechllm_ablation_de.pdf`,
  `speechllm_ablation_de.png`, `speechllm_ablation_ja.pdf`,
  `speechllm_ablation_ja.png`, `speechllm_ablation_zh_tau_compare.pdf`,
  `speechllm_ablation_zh_tau_compare.png`

This package plots the En-De rows used in the paper-facing Speech LLM ablation
and an additional En-Ja companion figure for inspection.
The old `main_result_tagged.pdf` RASST line is renamed to
`RASST (LLM-generated TM SFT)`, while the current `RASST` line comes from the
matching-language main result in `new_main_result_tagged.pdf`. The blue
horizontal reference is the oracle-term full-context offline upper bound from
the current main-result TSV.

The En-Zh tau-comparison companion keeps five lines only: `Offline ST`,
`Offline + GT Terms`, `InfiniSST`, `RASST (tau=0.78)`, and
`RASST (tau=0.0)`. Its tau=0.0 rows come from the verified ACL tagged raw
comparison TSV produced on 2026-05-26.

Original provenance:

- Legacy four-line figure data:
  `../../../../../../simuleval/reports/20260521_tagged_acl_main_results_fourline_llmgen_rasst_data.tsv`
- Legacy analysis manifest:
  `../../../../../../simuleval/manifests/2026/05/20260521T0859__analysis__tagged_acl_main_results_llmgen_rasst_figure.json`
- Current main-result data:
  `../figure_01_main_result_tagged/data.tsv`
- En-Zh tau=0.0 comparison data:
  `../../../../../../simuleval/reports/20260526_tagged_acl_zh_tau000_vs_main_rasst.tsv`

Regenerate the local snapshot from this directory:

```bash
python plot_figure_06_speechllm_ablation.py
```

Regenerate and update the paper-facing figure:

```bash
python plot_figure_06_speechllm_ablation.py --update-paper
```

Regenerate the En-Ja companion figure and copy it to `latex/figures/`:

```bash
python plot_figure_06_speechllm_ablation.py \
  --data data_ja.tsv \
  --out-prefix speechllm_ablation_ja \
  --paper-stem speechllm_ablation_ja \
  --x-right-pad 25 \
  --update-paper
```

Regenerate the En-Zh five-line tau comparison and copy it to `latex/figures/`:

```bash
python build_data_zh_tau_compare.py
python plot_figure_06_speechllm_ablation.py \
  --data data_zh_tau_compare.tsv \
  --out-prefix speechllm_ablation_zh_tau_compare \
  --paper-stem speechllm_ablation_zh_tau_compare \
  --update-paper
```
