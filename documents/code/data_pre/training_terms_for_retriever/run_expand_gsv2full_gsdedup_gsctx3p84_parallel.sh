#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
SCRIPT_PATH="${REPO_ROOT}/documents/code/data_pre/training_terms_for_retriever/expand_gigaspeech_context_3p84.py"

INPUT_JSONL="${INPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl}"
AUDIO_OUTPUT_DIR="${AUDIO_OUTPUT_DIR:-/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_gsctx3p84}"
WIKI_AUDIO_OUTPUT_DIR="${WIKI_AUDIO_OUTPUT_DIR:-${AUDIO_OUTPUT_DIR}/wiki_synth}"
STATS_JSON="${STATS_JSON:-/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84_stats.json}"

NUM_SHARDS="${NUM_SHARDS:-8}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_JSONL%.jsonl}_shards}"
LOG_DIR="${LOG_DIR:-/mnt/gemini/data1/jiaxuanluo/logs/gsctx3p84_shards}"
OVERWRITE_AUDIO="${OVERWRITE_AUDIO:-false}"
DRY_RUN="${DRY_RUN:-false}"

if pgrep -af "expand_gigaspeech_context_3p84.py .*--output ${OUTPUT_JSONL}" >/dev/null 2>&1; then
  echo "[ERROR] Another expand process is already writing ${OUTPUT_JSONL}." >&2
  echo "[ERROR] Stop it before running the parallel wrapper." >&2
  exit 2
fi

mkdir -p "${SHARD_DIR}" "${LOG_DIR}"

echo "[PARALLEL-GSCTX3P84] input=${INPUT_JSONL}"
echo "[PARALLEL-GSCTX3P84] output=${OUTPUT_JSONL}"
echo "[PARALLEL-GSCTX3P84] shards=${NUM_SHARDS} parallel_jobs=${PARALLEL_JOBS}"
echo "[PARALLEL-GSCTX3P84] shard_dir=${SHARD_DIR}"
echo "[PARALLEL-GSCTX3P84] audio_output_dir=${AUDIO_OUTPUT_DIR}"
echo "[PARALLEL-GSCTX3P84] wiki_audio_output_dir=${WIKI_AUDIO_OUTPUT_DIR}"

run_one() {
  local sid="$1"
  local shard_tag
  shard_tag="$(printf '%02d' "${sid}")"
  local shard_output="${SHARD_DIR}/part_${shard_tag}.jsonl"
  local shard_stats="${SHARD_DIR}/part_${shard_tag}_stats.json"
  local shard_log="${LOG_DIR}/expand_ctx3p84_part_${shard_tag}.log"

  local extra_args=()
  if [ "${OVERWRITE_AUDIO}" = "true" ]; then
    extra_args+=(--overwrite-audio)
  fi
  if [ "${DRY_RUN}" = "true" ]; then
    extra_args+=(--dry-run)
  fi

  echo "[PARALLEL-GSCTX3P84] start shard ${sid}/${NUM_SHARDS} log=${shard_log}"
  python "${SCRIPT_PATH}" \
    --input "${INPUT_JSONL}" \
    --output "${shard_output}" \
    --audio-output-dir "${AUDIO_OUTPUT_DIR}" \
    --wiki-audio-output-dir "${WIKI_AUDIO_OUTPUT_DIR}" \
    --stats-json "${shard_stats}" \
    --old-chunk-sec 1.92 \
    --new-chunk-sec 3.84 \
    --stride-sec 0.96 \
    --include-mode overlap \
    --num-shards "${NUM_SHARDS}" \
    --shard-id "${sid}" \
    "${extra_args[@]}" \
    > "${shard_log}" 2>&1
}

running=0
for sid in $(seq 0 $((NUM_SHARDS - 1))); do
  run_one "${sid}" &
  running=$((running + 1))
  if [ "${running}" -ge "${PARALLEL_JOBS}" ]; then
    wait -n
    running=$((running - 1))
  fi
done
wait

python - "${SHARD_DIR}" "${NUM_SHARDS}" "${OUTPUT_JSONL}" "${STATS_JSON}" "${AUDIO_OUTPUT_DIR}" "${WIKI_AUDIO_OUTPUT_DIR}" <<'PY'
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

metadata_keys = {
    "input",
    "output",
    "audio_output_dir",
    "wiki_audio_output_dir",
    "old_chunk_sec",
    "new_chunk_sec",
    "stride_sec",
    "include_mode",
    "num_shards",
    "shard_id",
    "dry_run",
    "write_empty_groups",
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
PY

echo "[PARALLEL-GSCTX3P84] DONE"
