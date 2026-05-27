# Medicine zh No-RAG Baseline Completion On Aries

## Hypothesis

Aries has enough idle A6000 pairs to finish the remaining `zh` no-RAG medicine
baseline faster than waiting for PSC V100 queue slots.

## Background / Motivation

PSC jobs for `medicine zh lm=1,3` have not reached usable resources.  The
remaining `zh` baseline gaps are:

- `lm=1`, samples `404 545006 596001 605000 606`
- `lm=3`, samples `404 545006 596001 605000 606`
- `lm=4`, samples `404 545006 596001 606`

`lm=4 sample=605000` is intentionally not rerun here.  It already exists from
the Taurus no-RAG vLLM override probe:

```text
documents/code/simuleval/manifests/2026/05/20260523T2110__simuleval__medicine_norag_vllm_override_probe_taurus.json
/mnt/gemini/data1/jiaxuanluo/medicine_norag_vllm_override_probe_20260523
```

## What changed vs baseline

- Reuses the existing batched medicine no-RAG launcher unchanged for decoding:
  `documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh`.
- Runs three independent `zh` settings with new output directories under
  `/mnt/gemini/data1/jiaxuanluo`.
- Uses short `/dev/shm` temp dirs to avoid vLLM IPC path length failures.
- Uses Triton/Torch/CUDA/XDG cache dirs under `/mnt/gemini/data1` to avoid the
  full `/mnt/data7` cache path on Aries.
- Runs hard-manual StreamLAAL term eval and miss export after each successful
  generation run.

## Expected metrics

Use the per-setting hard-manual StreamLAAL outputs:

```text
eval_results_streamlaal_term.hard_llm_manual_check.tsv
term_misses.hard_llm_manual_check.zh_lm*.tsv
term_miss_summary.hard_llm_manual_check.zh_lm*.tsv
```

For baseline analysis, prioritize `TERM_ACC` and the exported miss TSVs.
Raw SimulEval LAAL in `eval_results.tsv` is not the final latency metric for
this medicine hard-term analysis.

## Verdict

Planned as an Aries detached direct run.  Success requires:

- `lm=1` and `lm=3` timing rows show 5 samples completed.
- `lm=4` timing row shows 4 samples completed, and its sample map does not
  contain `605000`.
- `instances.log`, hard-manual StreamLAAL TSV, and hard-manual miss TSV exist
  for each new setting.

