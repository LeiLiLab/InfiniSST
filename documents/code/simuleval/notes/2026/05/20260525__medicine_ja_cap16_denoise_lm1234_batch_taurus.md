# SimulEval: JA Cap16-Denoise Medicine Hardraw Batch Taurus

## Hypothesis
The completed Japanese cap16-denoise short-tag SLM should recover medicine hardraw terminology when supplied HN1024/tau0.78 retrieved chunks, with short output tags stripped before scoring.

## Background / Motivation
The Japanese SLM was trained from `20260525T1506__data_prepare__ja_cap16_denoise_budget_ttag` and W&B run `wkoonqux`. This readout is the medicine-focused batch eval requested for `lm=1,2,3,4`.

## What changed vs baseline
- Model: `/mnt/data1/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf`.
- Dataset: medicine hardraw five-sample readout for Japanese.
- Retriever: HN1024 with score threshold `0.78`.
- Cache policy: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Prompt policy: `rag_prompt_policy=given_chunks`.
- Decode cap: `max_new_tokens=lm*40`, so `40,80,120,160` for `lm=1,2,3,4`.
- Empty term map policy: `omit`.
- Output scoring strips `<t>...</t>` short tags with `strip_output_tags=term_t`.
- Initial Taurus GPU pairs: `2,3;4,5`, two LMs per wave. At `2026-05-25T19:54:15Z`, idle pairs `0,1;6,7` were used to split out `lm=3,4`.

## Expected metrics
The primary check is `TERM_ACC` on the fixed hardraw medicine glossary denominator, with `BLEU`, `StreamLAAL`, `StreamLAAL_CA`, `REAL_TERM_ADOPT`, and `TERM_FCR` as supporting readouts from each `eval_results.tsv`.

## Verdict
Submitted detached on Taurus at `2026-05-25T19:24:06Z` as PID `3679891`. Pre-submit GPU check showed all eight GPUs idle at `2 MiB, 0%`.

Startup verified at `2026-05-25T19:26:18Z`: first wave `lm=1` and `lm=2` loaded vLLM from the Taurus local HF cache, initialized TP=2 engines on GPU pairs `2,3` and `4,5`, loaded the HN1024 retriever over 212 glossary terms, and began decoding with fixed max-new-token policies `{1: 40}` and `{2: 80}`.

At `2026-05-25T19:35:12Z`, first wave was still running with `lm=1` at step `470` and `lm=2` at step `343`; no `eval_results.tsv` had landed yet. A detached completion watcher was started as PID `3776658` and will notify on `.success` or process exit.

At `2026-05-25T19:54:15Z`, the old parent queue process `3679891` and watcher `3776658` were stopped so that it would not later duplicate `lm=3,4`. The already running `lm=1`/`lm=2` child evals were left alive as orphaned shell/Python jobs. A split partial runner was started as PID `3935533` with `LMS="3 4"`, `GPU_PAIRS_CSV="0,1;6,7"`, `SKIP_PREPARE=1`, `SKIP_GLOBAL_MERGE=1`, and `SKIP_SUCCESS_MARKER=1`. Startup was verified by vLLM model load and active `[STEP]` logs for both `lm=3` and `lm=4`. A replacement file-level completion watcher was started as PID `3949195`; it waits for all four per-LM summary TSVs before merging `summary_medicine_hardraw_ja_lm1_lm2_lm3_lm4.tsv` and writing `.success`.

Completed successfully at `2026-05-25T20:31:59Z`; all four `eval_results.tsv` files, per-LM summaries, and the merged summary are present. Each `instances.log` and `instances.strip_term.log` has 5 rows.

| lm | max_new | BLEU | StreamLAAL | StreamLAAL_CA | TERM_ACC | TERM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 18.4229 | 1474.4801 | 1650.9138 | 0.7577 | 491/648 |
| 2 | 80 | 24.6735 | 2186.5425 | 1767.1576 | 0.8102 | 525/648 |
| 3 | 120 | 26.8288 | 2781.0621 | 1782.7240 | 0.8380 | 543/648 |
| 4 | 160 | 28.6781 | 3199.2583 | 1708.7238 | 0.8349 | 541/648 |

- Launcher: `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/launchers/2026/05/20260525__medicine_ja_cap16_denoise_lm1234_batch_taurus.sh`
- Output base: `/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus`
- Log root: `/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus`
- PID file: `/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/direct_eval.pid`
- Completion watcher PID file: `/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/eval_completion_watch.pid`
- Split lm3/lm4 PID file: `/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/direct_eval_lm34.pid`
- Replacement completion watcher PID file: `/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/eval_completion_watch_split.pid`
- Merged summary: `/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/__summary__/summary_medicine_hardraw_ja_lm1_lm2_lm3_lm4.tsv`
- Manifest: `/home/jiaxuanluo/InfiniSST/documents/code/simuleval/manifests/2026/05/20260525T1840__simuleval__medicine_ja_cap16_denoise_lm1234_batch_taurus.json`
