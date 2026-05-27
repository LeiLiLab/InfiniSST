## Hypothesis

Serial SimulEval readouts with `max_cache_chunks=30`, `keep_cache_chunks=30`, `empty_term_map_policy=omit`, `system_prompt_style=given_chunks`, and `max_new_tokens=40*lm` provide a cleaner paper-facing comparison than mixed batch/serial artifacts for the current En-De and En-Ja RASST main-result rows.

## Background / Motivation

The current main-result TSV contains a mixture of serial and batch RASST readouts for En-De/En-Ja and for tagged ACL / medicine hardraw. The user requested rerunning the main-result RASST rows in serial mode for `lm=1,2,3,4`, first on tagged ACL and then on medicine.

Aries was checked before launch and was not idle: all 8 GPUs were running Python/vLLM-style processes at roughly 37GB and 100% utilization. Taurus was idle, so this launcher fills Taurus with four concurrent 2-GPU serial jobs and enforces stage ordering.

`/mnt/gemini/data1` was also checked before launch and had only about 91GB free while reporting 100% usage, so output and logs are placed on Taurus-local `/mnt/data1` for this rerun.

## What changed vs baseline

- Re-run RASST only for `lang in {de, ja}` and `lm in {1,2,3,4}`.
- Tagged ACL stage runs before the medicine hardraw stage.
- Serial driver: `documents/code/simuleval/eval_density_unified.sh`.
- Cache policy: `max_cache_chunks=30`, `keep_cache_chunks=30`.
- Prompt policy: `system_prompt_style=given_chunks` and `empty_term_map_policy=omit`.
- Decode budget: `max_new_tokens=40*lm`.
- Retriever: HN1024 checkpoint with `top_k=10`, `tau=0.78`, timeline lookback `1.92s`.
- Output tags stripped with `term_t` so both `<term>` and `<t>` style wrappers are handled.

## Expected metrics

Expected output is one validated `eval_results.tsv`, `instances.log`, and `instances.strip_term.log` per `(dataset, lang, lm)` with 5 talk/sample rows. Metrics are not selected here; this is a readout rerun.

## Verdict

Submitted. Await per-task `*.done` or `*.failed` markers under the log root.
