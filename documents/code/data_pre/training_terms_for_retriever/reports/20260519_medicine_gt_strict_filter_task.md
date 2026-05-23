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

## Post-Eval

Do not use raw `LAAL` / `AL` from SimulEval `scores.tsv` as the final latency
number. For these long medicine talks, run the separate StreamLAAL post-eval.

Extra tools needed:

```bash
export FBK_FAIRSEQ_ROOT="/path/to/FBK-fairseq"
export STREAM_LAAL_TOOL="${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
export MWERSEGMENTER_ROOT="/path/to/mwerSegmenter"
export PATH="${MWERSEGMENTER_ROOT}:${PATH}"
```

For each completed `(lang, lm)`, set:

```bash
LANG=zh
LM=2
OUT_DIR="/path/to/output_dir_from_timing_tsv"
COMBINED_DIR="$OUTPUT_BASE_OVERRIDE/$LANG/__medicine_inputs__/combined"
```

Run StreamLAAL + TERM_ACC:

```bash
"${CONDA_PREFIX}/bin/python" documents/code/offline_sst_eval/offline_streamlaal_eval.py \
  --mode acl6060 \
  --instances-log "$OUT_DIR/instances.log" \
  --lang-code "$LANG" \
  --ref-file "$COMBINED_DIR/medicine5.ref.$LANG.sentences.txt" \
  --source-file "$COMBINED_DIR/medicine5.source_text.en.sentences.txt" \
  --audio-yaml "$COMBINED_DIR/medicine5.audio.yaml" \
  --glossary-acl6060 "$OUTPUT_BASE_OVERRIDE/strict_fixed_medicine_glossary.from_outputs_v2_terms.json" \
  --fbk-fairseq-root "$FBK_FAIRSEQ_ROOT" \
  --term-fcr-policy source_ref_negative_sentence \
  --output-tsv "$OUT_DIR/eval_results_streamlaal_term.tsv" \
  --output-log "$OUT_DIR/post_eval_streamlaal_term_full.log" \
  --work-dir "$OUT_DIR/work_streamlaal_term" \
  --term-mismatch-examples 20
```

Run full miss extraction:

```bash
"${CONDA_PREFIX}/bin/python" documents/code/simuleval/export_streamlaal_term_misses.py \
  --instances-log "$OUT_DIR/instances.log" \
  --reference "$COMBINED_DIR/medicine5.ref.$LANG.sentences.txt" \
  --source-reference "$COMBINED_DIR/medicine5.source_text.en.sentences.txt" \
  --audio-yaml "$COMBINED_DIR/medicine5.audio.yaml" \
  --glossary "$OUTPUT_BASE_OVERRIDE/strict_fixed_medicine_glossary.from_outputs_v2_terms.json" \
  --lang-code "$LANG" \
  --stream-laal-tool "$STREAM_LAAL_TOOL" \
  --mwersegmenter-root "$MWERSEGMENTER_ROOT" \
  --output-misses "$OUT_DIR/term_misses.${LANG}_lm${LM}.tsv" \
  --output-summary "$OUT_DIR/term_miss_summary.${LANG}_lm${LM}.tsv" \
  --output-normalized-glossary "$OUT_DIR/strict_fixed_medicine_glossary.streamlaal_dict.json"
```

TERM_ACC means exact target translation hit among sentence-level occurrences
where both the English source term and target reference translation are present.
The miss TSV is the main file for manual strict-term filtering.

For this no-RAG baseline, ignore TERM_ADOPTION / REAL_TERM_ADOPT and do not use
TERM_FCR as the selection signal; there is no retrieval term_map. Use
`TERM_ACC`, `term_misses`, and manual labels.

Jiaxuan's completed sanity check for `zh/lm=2`:

```text
StreamLAAL = 1509.43 ms
StreamLAAL_CA = 2024.97 ms
TERM_ACC = 0.6343 = 1426 / 2248
miss occurrences = 822
unique missed term-translations = 511
```

## Check Outputs

Main output files:

```text
$OUTPUT_BASE/strict_fixed_medicine_glossary.from_outputs_v2_terms.json
$OUTPUT_BASE/timing.tsv
$OUTPUT_BASE/hypotheses.tsv
$OUTPUT_BASE/<lang>/<run_dir>/instances.log
$OUTPUT_BASE/<lang>/<run_dir>/runtime_omni_vllm_rag_v4_*.jsonl
$OUTPUT_BASE/<lang>/<run_dir>/eval_results_streamlaal_term.tsv
$OUTPUT_BASE/<lang>/<run_dir>/term_misses.<lang>_lm<lm>.tsv
$OUTPUT_BASE/<lang>/<run_dir>/term_miss_summary.<lang>_lm<lm>.tsv
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
eval_results_streamlaal_term.tsv for each completed setting
term_misses.<lang>_lm<lm>.tsv for each completed setting
term_miss_summary.<lang>_lm<lm>.tsv for each completed setting
all non-empty instances.log paths
the launcher you used on PSC
stdout/stderr logs
```

Manual review should start from `term_misses.<lang>_lm<lm>.tsv`, not from raw
BLEU/LAAL. The review target is baseline-missed strict medicine terms.

## Reminder

This filtered set is a **hard-term diagnostic set**, not a neutral medicine
benchmark. In paper text, say the final strict terms are selected using
streaming no-RAG baseline failure.
