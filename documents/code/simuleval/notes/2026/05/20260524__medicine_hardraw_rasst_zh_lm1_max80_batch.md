## Hypothesis

Rerunning the zh medicine hardraw RASST lm=1 batch with `max_new_tokens=80` should improve completeness versus the earlier lm=1 readout that used the default 40-token cap.

## Background / Motivation

The medicine main-result figure currently reads the zh RASST hardraw lm=1 value from the five-sample batch run under `20260524T0242`. That run used the legacy default decode cap. This run keeps the same model, retriever, glossary, and five-sample combined input, but changes the decode cap to 80.

## What changed vs baseline

- language: `zh`
- dataset: medicine hardraw five samples `404 545006 596001 605000 606`
- lm: `1`
- max new tokens: `80`
- runtime glossary: hard medicine raw manual glossary
- metric denominator: fixed hard medicine raw manual glossary
- retriever: HN1024 `lh1b88kw`, tau `0.78`, top-k `10`
- Speech LLM: zh New V9 RASST checkpoint
- launcher: medicine five-sample batch launcher, one lm process for all five samples

## Expected metrics

The output should contain one `eval_results.tsv` row for zh lm=1 with BLEU, StreamLAAL, TERM_ACC, REAL_TERM_ADOPT, and TERM_FCR. After success, the medicine main-result figure should prefer this lm=1 max80 file over the older lm=1 file.

## Verdict

SUCCESS AS A DIAGNOSTIC RUN, REJECTED FOR MAIN RESULT. The five-sample zh medicine hardraw RASST lm=1 batch completed with `max_new_tokens=80`, but the main-result TSV/PDF were later reverted to the earlier max40 lm=1 result because this run degraded BLEU and TERM_ACC.

- eval results: `/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T1748_medicine_rasst_zh_lm1_max80_sharedaudio_batch/zh/dmedhard5_new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_max80_raw_lm1_k10_th0.78_ghard_medicine_glossary_raw_llm_judge_manual_zh215_unique212_ppmedicine5_hardraw/eval_results.tsv`
- main TSV: `documents/code/simuleval/reports/20260524_main_result_data.tsv`
- medicine figure: `documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/medicine_main_result.pdf`
- BLEU: 32.0754
- StreamLAAL: 1224.08
- TERM_ACC: 0.7459 (502/673)
- REAL_TERM_ADOPT: 0.8481 (449/528)
- TERM_FCR: 0.3453

Note: this max80 rerun is worse than the previous lm=1 max40 readout, likely due to long-output degeneration observed in sample 605000. The selected main-result row is tracked by `20260524T2149__analysis__medicine_main_result_revert_zh_rasst_lm1_max40`.
