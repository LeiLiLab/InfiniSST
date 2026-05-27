#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/build_variable_gigaspeech_context.py"

INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84.jsonl}"
AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx0p96_1p92_2p88_3p84}"
WIKI_AUDIO_OUTPUT_DIR="${WIKI_AUDIO_OUTPUT_DIR:-${AUDIO_OUTPUT_DIR}/wiki_synth}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84_stats.json}"
DIAG_JSON="${DIAG_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84_diag.json}"

DURATION_SECS="${DURATION_SECS:-0.96 1.92 2.88 3.84}"
DURATION_ASSIGNMENT="${DURATION_ASSIGNMENT:-balance_rows}"
NUM_SHARDS="${NUM_SHARDS:-8}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
START_SHARD="${START_SHARD:-0}"
END_SHARD="${END_SHARD:-$((NUM_SHARDS - 1))}"
SKIP_COMPLETED_SHARDS="${SKIP_COMPLETED_SHARDS:-false}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_JSONL%.jsonl}_shards}"
LOG_DIR="${LOG_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/varctx_shards}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"
DRY_RUN="${DRY_RUN:-false}"
REUSE_OLD_AUDIO_FOR_1P92="${REUSE_OLD_AUDIO_FOR_1P92:-true}"
COPY_UNEXPANDABLE_GS="${COPY_UNEXPANDABLE_GS:-true}"
WIKI_EXPAND_FAILURE_POLICY="${WIKI_EXPAND_FAILURE_POLICY:-fallback}"
RUN_DIAG="${RUN_DIAG:-true}"
DIAG_TARGET_FRAC="${DIAG_TARGET_FRAC:-}"
DIAG_FRAC_TOLERANCE="${DIAG_FRAC_TOLERANCE:-}"
DIAG_NO_FAIL="${DIAG_NO_FAIL:-false}"

if pgrep -af "build_variable_gigaspeech_context.py .*--output ${OUTPUT_JSONL}" >/dev/null 2>&1; then
  echo "[ERROR] Another variable-context process is already writing ${OUTPUT_JSONL}." >&2
  exit 2
fi

mkdir -p "${SHARD_DIR}" "${LOG_DIR}"

if [ "${START_SHARD}" -lt 0 ] || [ "${END_SHARD}" -ge "${NUM_SHARDS}" ] || [ "${START_SHARD}" -gt "${END_SHARD}" ]; then
  echo "[ERROR] Invalid shard range START_SHARD=${START_SHARD} END_SHARD=${END_SHARD} NUM_SHARDS=${NUM_SHARDS}" >&2
  exit 2
fi

echo "[PARALLEL-VARCTX] input=${INPUT_JSONL}"
echo "[PARALLEL-VARCTX] output=${OUTPUT_JSONL}"
echo "[PARALLEL-VARCTX] durations=${DURATION_SECS} assignment=${DURATION_ASSIGNMENT}"
echo "[PARALLEL-VARCTX] shards=${NUM_SHARDS} shard_range=${START_SHARD}-${END_SHARD} parallel_jobs=${PARALLEL_JOBS} skip_completed=${SKIP_COMPLETED_SHARDS}"
echo "[PARALLEL-VARCTX] shard_dir=${SHARD_DIR}"
echo "[PARALLEL-VARCTX] audio_output_dir=${AUDIO_OUTPUT_DIR}"
echo "[PARALLEL-VARCTX] wiki_audio_output_dir=${WIKI_AUDIO_OUTPUT_DIR}"

run_one() {
  local sid="$1"
  local shard_tag
  shard_tag="$(printf '%02d' "${sid}")"
  local shard_output="${SHARD_DIR}/part_${shard_tag}.jsonl"
  local shard_stats="${SHARD_DIR}/part_${shard_tag}_stats.json"
  local shard_log="${LOG_DIR}/build_varctx_part_${shard_tag}.log"

  local extra_args=()
  if [ "${OVERWRITE_AUDIO}" = "true" ]; then
    extra_args+=(--overwrite-audio)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    extra_args+=(--dry-run)
  fi
  if [ "${REUSE_OLD_AUDIO_FOR_1P92}" = "true" ]; then
    extra_args+=(--reuse-old-audio-for-1p92)
  else
    extra_args+=(--no-reuse-old-audio-for-1p92)
  fi
  if [ "${COPY_UNEXPANDABLE_GS}" = "true" ]; then
    extra_args+=(--copy-unexpandable-gs)
  else
    extra_args+=(--no-copy-unexpandable-gs)
  fi
  extra_args+=(--wiki-expand-failure-policy "${WIKI_EXPAND_FAILURE_POLICY}")

  echo "[PARALLEL-VARCTX] start shard ${sid}/${NUM_SHARDS} log=${shard_log}"
  python "${SCRIPT_PATH}" \
    --input "${INPUT_JSONL}" \
    --output "${shard_output}" \
    --audio-output-dir "${AUDIO_OUTPUT_DIR}" \
    --wiki-audio-output-dir "${WIKI_AUDIO_OUTPUT_DIR}" \
    --stats-json "${shard_stats}" \
    --old-chunk-sec 1.92 \
    --stride-sec 0.96 \
    --duration-secs "${DURATION_SECS}" \
    --duration-assignment "${DURATION_ASSIGNMENT}" \
    --include-mode overlap \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${sid}" \
    "${extra_args[@]}" \
    > "${shard_log}" 2>&1
}

running=0
for sid in $(seq "${START_SHARD}" "${END_SHARD}"); do
  shard_tag="$(printf '%02d' "${sid}")"
  shard_output="${SHARD_DIR}/part_${shard_tag}.jsonl"
  shard_stats="${SHARD_DIR}/part_${shard_tag}_stats.json"
  if [ "${SKIP_COMPLETED_SHARDS}" = "true" ] && [ -s "${shard_output}" ] && [ -s "${shard_stats}" ]; then
    echo "[PARALLEL-VARCTX] skip completed shard ${sid}/${NUM_SHARDS} output=${shard_output}"
    continue
  fi
  run_one "${sid}" &
  running=$((running + 1))
  if [ "${running}" -ge "${PARALLEL_JOBS}" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

python - "${SHARD_DIR}" "${NUM_SHARDS}" "${OUTPUT_JSONL}" "${STATS_JSON}" "${AUDIO_OUTPUT_DIR}" "${WIKI_AUDIO_OUTPUT_DIR}" "${DURATION_SECS}" "${DURATION_ASSIGNMENT}" <<'PY'
import json
import os
import sys
from pathlib import Path

shard_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
output_jsonl = Path(sys.argv[3])
stats_json = Path(sys.argv[4])
audio_output_dir = sys.argv[5]
wiki_audio_output_dir = sys.argv[6]
duration_secs = [float(x) for x in sys.argv[7].replace(",", " ").split()]
duration_assignment = sys.argv[8]

metadata_keys = {
    "input",
    "output",
    "audio_output_dir",
    "wiki_audio_output_dir",
    "old_chunk_sec",
    "duration_secs",
    "duration_tags",
    "stride_sec",
    "include_mode",
    "duration_assignment",
    "num_shards",
    "shard_id",
    "dry_run",
    "write_empty_groups",
    "reuse_old_audio_for_1p92",
    "stats_json",
}

tmp_output = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
tmp_output.parent.mkdir(parents=True, exist_ok=True)

merged = {}
shard_stats = []
with open(tmp_output, "w", encoding="utf-8") as fout:
    for sid in range(num_shards):
        tag = f"{sid:02d}"
        part = shard_dir / f"part_{tag}.jsonl"
        stats_path = shard_dir / f"part_{tag}_stats.json"
        if not part.is_file():
            raise FileNotFoundError(part)
        if not stats_path.is_file():
            raise FileNotFoundError(stats_path)
        with open(part, "r", encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
        with open(stats_path, "r", encoding="utf-8") as fin:
            stats = json.load(fin)
        shard_stats.append({"shard_id": sid, "stats_path": str(stats_path)})
        for key, value in stats.items():
            if key in metadata_keys:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value

os.replace(tmp_output, output_jsonl)
merged.update(
    {
        "input": "sharded:" + str(shard_dir),
        "output": str(output_jsonl),
        "audio_output_dir": audio_output_dir,
        "wiki_audio_output_dir": wiki_audio_output_dir,
        "duration_secs": duration_secs,
        "duration_tags": [str(x).rstrip("0").rstrip(".").replace(".", "p") for x in duration_secs],
        "duration_assignment": duration_assignment,
        "num_shards": num_shards,
        "shard_stats": shard_stats,
    }
)
stats_json.parent.mkdir(parents=True, exist_ok=True)
with open(stats_json, "w", encoding="utf-8") as fout:
    json.dump(merged, fout, indent=2, ensure_ascii=False, sort_keys=True)

print(f"[MERGE] output={output_jsonl}")
print(f"[MERGE] stats={stats_json}")
print(f"[MERGE] written_total_rows={merged.get('written_total_rows')}")
for tag in merged.get("duration_tags", []):
    print(f"[MERGE] duration_row_count_{tag}={merged.get('duration_row_count_' + tag)}")
PY

if [ "${RUN_DIAG}" = "true" ] && [ "${DRY_RUN}" != "true" ]; then
  diag_args=()
  if [ -n "${DIAG_TARGET_FRAC}" ]; then
    diag_args+=(--target-frac "${DIAG_TARGET_FRAC}")
  fi
  if [ -n "${DIAG_FRAC_TOLERANCE}" ]; then
    diag_args+=(--frac-tolerance "${DIAG_FRAC_TOLERANCE}")
  fi
  if [ "${DIAG_NO_FAIL}" = "true" ]; then
    diag_args+=(--no-fail)
  fi
  python "${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/diagnose_variable_context_jsonl.py" \
    --input "${OUTPUT_JSONL}" \
    --stats-json "${STATS_JSON}" \
    --expected-duration-secs "${DURATION_SECS}" \
    --report-json "${DIAG_JSON}" \
    "${diag_args[@]}"
fi

echo "[PARALLEL-VARCTX] DONE"
