## Hypothesis

The ESO DE/JA folders under `documents/code/train/term_train/reports/figures/eso_de_ja_results` contain the current InfiniSST medicine baseline outputs. Recomputing sentence-level BLEU, StreamLAAL, and TERM_ACC from those `instances.log` files should replace the older medicine hardraw InfiniSST rows in the main-result TSV and figure.

## Background / Motivation

The previous medicine hardraw baseline rows came from older no-RAG post-eval artifacts. The new ESO folders provide the current baseline predictions for DE and JA, with `seg960`, `seg1920`, `seg2880`, and `seg3840` corresponding to `lm=1`, `lm=2`, `lm=3`, and `lm=4`.

The file `test.source` records the five-talk evaluation order. Post-eval uses the existing medicine hardraw source/reference/audio/glossary lists so the denominator and StreamLAAL segmentation remain comparable with the main-result medicine setting.

## What changed vs baseline

- Added `documents/code/simuleval/src/update_medicine_baseline_from_eso_de_ja_20260525.py`.
- Recomputed DE/JA InfiniSST medicine hardraw baseline rows from the new ESO `instances.log` files.
- Used sentence-resegmented post-eval BLEU, not the talk-level `scores.tsv` BLEU, for the main-result TSV.
- Rebuilt the JA combined post-eval source/reference/audio lists from per-sample files in `test.source` order. The historical JA combined list under the NewV9 lm12 output was malformed: it started at `545006_v2.wav` and duplicated later talks, which produced inflated StreamLAAL and the wrong term denominator.
- Updated:
  - `documents/code/simuleval/reports/20260524_main_result_data.tsv`
  - `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/plot/figure_01_main_result_tagged/data.tsv`
  - `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/plot/figure_02_medicine_main_result/data.tsv`
  - `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/medicine_main_result.pdf`
  - `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/medicine_main_result.png`

## Expected metrics

The updated DE/JA InfiniSST rows should report sentence-level post-eval BLEU/StreamLAAL/TERM_ACC. `scores.tsv` is retained only as the original talk-level BLEU reference in each generated `eval_results.main.tsv`.

## Verdict

Success after the JA alignment fix. The canonical main-result TSV now uses the new ESO DE/JA InfiniSST baseline rows. The medicine main-result figure was regenerated and copied to the paper figure directory.
