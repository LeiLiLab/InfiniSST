# Medicine no-RAG Baseline Handoff

Date: 2026-05-23 UTC

## Goal

Run streaming Qwen3-Omni **no-RAG** baseline on the restored ESO medicine data.
Use the hypotheses to find terms that the no-RAG streaming model misses or
mistranslates.

This is hypothesis generation only. Do not use offline/full-context Qwen3-Omni
as the main filter.

Current split:

- Jiaxuan runs: `lang=zh`, `lm=2`, 5 samples.
- Jiaxing runs on PSC:
  - `lang=zh`, `lm=1 3 4`
  - `lang=de ja`, `lm=1 2 3 4`

Samples:

```text
404 545006 596001 605000 606
```

## Input Terms

Use only the new restored ESO output root:

```text
/home/jiaxingxu/rag-sst/eso-dataset/outputs_v2_abbrev_exact_match_abbrev_restored/test
```

Each sample has:

```text
sample_<id>_v2/full_sample_v2.json
```

Terms are under:

```python
full_sample["sentences"][i]["terms"][j]
```

Each term entry has:

```json
{
  "term": "...",
  "target_translations": {
    "zh": "...",
    "de": "...",
    "ja": "..."
  }
}
```

The launcher now auto-builds:

```text
$OUTPUT_BASE/strict_fixed_medicine_glossary.from_outputs_v2_terms.json
```

from the 5 samples above. It preserves distinct translation variants instead
of collapsing all rows with the same English term.

This is **not** the old term/glossary list from last month. It uses the current
restored output version, where `terms` have already gone through the
substring/exact-match restoration checks.

For Jiaxing task, the review universe is the **1123 unique English source
terms** from these current `terms` annotations. The larger entry count only
exists because the same English term can have multiple translation variants.

## Key Files

Run this launcher:

```text
documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh
```

Base no-RAG runner:

```text
documents/code/simuleval/rank16/baseline/bypass_simuleval_rank32_iter_0000452_hf_baseline_no_rag_sweep.sh
```

The script batches 5 samples into one run per `(lang, lm)` so the model is not
reloaded once per sample.

## PSC Environment

Use the packed `spaCyEnv` unless PSC already has an equivalent env with
`torch`, `vllm`, `transformers`, `simuleval`, `yaml`, and repo dependencies.

Packed env source:

```text
/mnt/gemini/home/jiaxuanluo/transfer_packages/spaCyEnv_20260518.tar.gz
```

Unpack on PSC scratch:

```bash
mkdir -p /path/to/envs/spaCyEnv
tar -xzf spaCyEnv_20260518.tar.gz -C /path/to/envs/spaCyEnv
/path/to/envs/spaCyEnv/bin/conda-unpack
```

Minimal check:

```bash
export CONDA_PREFIX=/path/to/envs/spaCyEnv
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import torch, transformers, vllm, simuleval, yaml
print("env ok")
PY
```

Use 2 GPUs per run. Put caches and outputs on PSC scratch:

```bash
export HF_HOME=/path/to/psc/scratch/hf
export TRANSFORMERS_CACHE=$HF_HOME
export VLLM_USE_V1=0
```

## Script Config

Set these in the sbatch script or environment:

```bash
export CONDA_PREFIX="/path/to/envs/spaCyEnv"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
ROOT_DIR_OVERRIDE="/path/to/InfiniSST"
PREP_PYTHON_OVERRIDE="${CONDA_PREFIX}/bin/python"
ESO_TEST_ROOT_OVERRIDE="/path/to/outputs_v2_abbrev_exact_match_abbrev_restored/test"
OUTPUT_BASE_OVERRIDE="/path/to/psc/scratch/medicine_norag_baseline_abbrev_restored_batched"
CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="0:1"
MODEL_ZH_OVERRIDE="/path/to/gigaspeech-zh-s_origin-bsz4"
MODEL_DE_OVERRIDE="/path/to/gigaspeech-de-s_origin-bsz4"
MODEL_JA_OVERRIDE="/path/to/gigaspeech-ja-s_origin-bsz4"
```

No external glossary path is needed for this task. Leaving the glossary override
unset makes the launcher build
`strict_fixed_medicine_glossary.from_outputs_v2_terms.json` from
`full_sample_v2.json`.

Keep these defaults unless told otherwise:

```bash
TARGET_SAMPLES_OVERRIDE="404 545006 596001 605000 606"
TERM_SOURCE_OVERRIDE="glossary_match"
GLOSSARY_SOURCE_FILTER_OVERRIDE="strict_fixed_medicine_glossary"
RAG_K2_VALUE_OVERRIDE="10"
```

## Run Commands

Run via Slurm or detached shell. Do not run long jobs in a foreground terminal.

Run remaining zh settings:

```bash
cd /path/to/InfiniSST
setsid bash -lc '
  export CONDA_PREFIX="/path/to/envs/spaCyEnv"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  LANGS_OVERRIDE="zh" \
  TARGET_LMS_OVERRIDE="1 3 4" \
  ROOT_DIR_OVERRIDE="/path/to/InfiniSST" \
  PREP_PYTHON_OVERRIDE="${CONDA_PREFIX}/bin/python" \
  ESO_TEST_ROOT_OVERRIDE="/path/to/outputs_v2_abbrev_exact_match_abbrev_restored/test" \
  MODEL_ZH_OVERRIDE="/path/to/gigaspeech-zh-s_origin-bsz4" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="0:1" \
  OUTPUT_BASE_OVERRIDE="/path/to/psc/scratch/medicine_norag_baseline_abbrev_restored_batched" \
  bash documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh
' > /path/to/psc/scratch/logs/medicine_norag_zh_lm134.out \
  2> /path/to/psc/scratch/logs/medicine_norag_zh_lm134.err < /dev/null &
echo $! > /path/to/psc/scratch/logs/medicine_norag_zh_lm134.pid
```

Run de/ja settings:

```bash
cd /path/to/InfiniSST
setsid bash -lc '
  export CONDA_PREFIX="/path/to/envs/spaCyEnv"
  export PATH="${CONDA_PREFIX}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  LANGS_OVERRIDE="de ja" \
  TARGET_LMS_OVERRIDE="1 2 3 4" \
  ROOT_DIR_OVERRIDE="/path/to/InfiniSST" \
  PREP_PYTHON_OVERRIDE="${CONDA_PREFIX}/bin/python" \
  ESO_TEST_ROOT_OVERRIDE="/path/to/outputs_v2_abbrev_exact_match_abbrev_restored/test" \
  MODEL_DE_OVERRIDE="/path/to/gigaspeech-de-s_origin-bsz4" \
  MODEL_JA_OVERRIDE="/path/to/gigaspeech-ja-s_origin-bsz4" \
  CUDA_VISIBLE_DEVICES_PHYSICAL_OVERRIDE_CSV="0:1" \
  OUTPUT_BASE_OVERRIDE="/path/to/psc/scratch/medicine_norag_baseline_abbrev_restored_batched" \
  bash documents/code/simuleval/launchers/2026/05/20260522__medicine_abbrev_restored_norag_streaming_batched_aries67.sh
' > /path/to/psc/scratch/logs/medicine_norag_deja_lm1234.out \
  2> /path/to/psc/scratch/logs/medicine_norag_deja_lm1234.err < /dev/null &
echo $! > /path/to/psc/scratch/logs/medicine_norag_deja_lm1234.pid
```

If PSC requires `sbatch`, put the same environment block inside the sbatch
script and request 2 GPUs.

## Check Outputs

Main output files:

```text
$OUTPUT_BASE/strict_fixed_medicine_glossary.from_outputs_v2_terms.json
$OUTPUT_BASE/timing.tsv
$OUTPUT_BASE/hypotheses.tsv
$OUTPUT_BASE/<lang>/<run_dir>/instances.log
$OUTPUT_BASE/<lang>/<run_dir>/runtime_omni_vllm_rag_v4_*.jsonl
```

Quick checks:

```bash
python - <<'PY'
import json, os
p = os.environ["OUTPUT_BASE_OVERRIDE"] + "/strict_fixed_medicine_glossary.from_outputs_v2_terms.json"
rows = json.load(open(p, encoding="utf-8"))
print("strict_fixed_terms_entries", len(rows))
print("unique_terms", len({r["term"].casefold() for r in rows}))
PY
cat $OUTPUT_BASE_OVERRIDE/timing.tsv
wc -l $OUTPUT_BASE_OVERRIDE/hypotheses.tsv
find $OUTPUT_BASE_OVERRIDE -name instances.log -size +0 -print
```

Expected `hypotheses.tsv`: one header plus one row per completed
`(lang, lm, sample)`.

Expected fixed-term review universe: `1123` unique English source terms from
the current restored `terms` annotations.

## What To Send Back

Send back:

```text
strict_fixed_medicine_glossary.from_outputs_v2_terms.json
timing.tsv
hypotheses.tsv
all non-empty instances.log paths
the launcher you used on PSC
stdout/stderr logs
```

Do not manually label terms yet. The next step is exact-match + manual review
against `target_translations` from the fixed sample terms.

## Reminder

This filtered set is a **hard-term diagnostic set**, not a neutral medicine
benchmark. In paper text, say the final strict terms are selected using
streaming no-RAG baseline failure.
