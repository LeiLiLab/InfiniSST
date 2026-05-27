#!/usr/bin/env bash
# Build de New V9 Speech LLM SFT data with an old-new_v3 source candidate gate.
#
# This launcher first extracts utterance-level noun/entity candidates from the
# original TSV text, then uses those candidates only as a type/phrase allowlist.
# GT evidence still comes from MFA exact source matching and OpenAI exact
# future-reference span rewrite in the downstream New V9 builder.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_ROOT="${OUT_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
OUT_NAME_PREFIX="${OUT_NAME_PREFIX_OVERRIDE:-speech_llm_new_v9_mfa_openai_npfilter_oldnewv3}"
OUT_DIR="${OUT_ROOT}/${OUT_NAME_PREFIX}_de_20260524"
LOG_DIR="${OUT_DIR}/logs"
CAND_DIR="${OUT_DIR}/source_candidates"
mkdir -p "${LOG_DIR}" "${CAND_DIR}"

SPACY_ENV="${SPACY_ENV_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
SPACY_PY="${SPACY_PY_OVERRIDE:-${SPACY_ENV}/bin/python}"
SPACY_MODEL="${SPACY_MODEL_OVERRIDE:-en_core_web_trf}"
GPU_DEVICES_CSV="${GPU_DEVICES_CSV_OVERRIDE:-6,7}"
SPACY_SHARDS="${SPACY_SHARDS_OVERRIDE:-2}"

INPUT_JSONL="${DATA_ROOT}/train_s_de_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"
INPUT_TSV="${DATA_ROOT}/train_xl_case_robust_asr-filtered_de_metricx-qe3.0_align.tsv"
CAND_PREFIX="${CAND_DIR}/source_candidates_de_oldnewv3_spacy"
CAND_JSONL="${CAND_DIR}/source_candidates_de_oldnewv3_spacy.jsonl"
BASE_SOURCE_GLOSSARY="${SOURCE_GLOSSARY_OVERRIDE:-/mnt/gemini/home/jiaxuanluo/eval_glossaries/p31_untrained_dev/wiki_p31_untrained_rank1000000_sample100000.json}"
MERGED_SOURCE_GLOSSARY="${OUT_DIR}/source_glossary_wiki100k_plus_de_oldnewv3_candidates.json"
BASE_BUILDER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260524__build_new_v9_mfa_openai_rewrite_oldnewv3_de_ja.sh"
EXTRACT_SCRIPT="${ROOT_DIR}/retriever/gigaspeech/extract_ner_candidates_v4.py"

for p in "${SPACY_PY}" "${INPUT_JSONL}" "${INPUT_TSV}" "${BASE_SOURCE_GLOSSARY}" "${BASE_BUILDER}" "${EXTRACT_SCRIPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] GPU_DEVICES_CSV=${GPU_DEVICES_CSV}"
echo "[INFO] SPACY_SHARDS=${SPACY_SHARDS} SPACY_MODEL=${SPACY_MODEL}"
echo "[INFO] INPUT_JSONL=${INPUT_JSONL}"
echo "[INFO] INPUT_TSV=${INPUT_TSV}"
echo "[INFO] BASE_SOURCE_GLOSSARY=${BASE_SOURCE_GLOSSARY}"

IFS=',' read -r -a GPU_DEVICES <<< "${GPU_DEVICES_CSV}"
if (( ${#GPU_DEVICES[@]} < SPACY_SHARDS )); then
  echo "[ERROR] Need at least SPACY_SHARDS visible GPU ids in GPU_DEVICES_CSV." >&2
  exit 2
fi

if [[ "${FORCE_OVERWRITE:-0}" == "1" ]]; then
  rm -f "${CAND_PREFIX}"_gpu*.jsonl "${CAND_JSONL}"
fi

if [[ ! -s "${CAND_JSONL}" ]]; then
  echo "[STAGE P0] Extract old-new_v3 noun/entity source candidate allowlist"
  pids=()
  for ((shard=0; shard<SPACY_SHARDS; shard++)); do
    gpu="${GPU_DEVICES[$shard]}"
    log_file="${LOG_DIR}/stageP0_spacy_candidates_de_shard${shard}.log"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PYTHONNOUSERSITE=1
      "${SPACY_PY}" "${EXTRACT_SCRIPT}" \
        --input-gt "${INPUT_JSONL}" \
        --input-tsv "${INPUT_TSV}" \
        --output-jsonl "${CAND_PREFIX}.jsonl" \
        --spacy-model "${SPACY_MODEL}" \
        --gpu-id "${shard}" \
        --total-gpus "${SPACY_SHARDS}"
    ) > "${log_file}" 2>&1 &
    pids+=("$!")
    echo "[INFO] source-candidate shard=${shard} gpu=${gpu} pid=${pids[-1]} log=${log_file}"
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  : > "${CAND_JSONL}"
  for ((shard=0; shard<SPACY_SHARDS; shard++)); do
    shard_file="${CAND_PREFIX}_gpu${shard}.jsonl"
    if [[ ! -s "${shard_file}" ]]; then
      echo "[ERROR] Missing source candidate shard: ${shard_file}" >&2
      exit 4
    fi
    cat "${shard_file}" >> "${CAND_JSONL}"
  done
fi

"${SPACY_PY}" - "${CAND_JSONL}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
p = Path(sys.argv[1])
rows = 0
cands = 0
hist = Counter()
for line in p.open("r", encoding="utf-8"):
    if not line.strip():
        continue
    rows += 1
    obj = json.loads(line)
    items = obj.get("ner_candidates") or []
    cands += len(items)
    hist[len(items)] += 1
print(json.dumps({
    "source_candidate_jsonl": p.as_posix(),
    "rows": rows,
    "candidates": cands,
    "avg_candidates_per_row": cands / rows if rows else 0.0,
    "empty_rows": hist.get(0, 0),
}, ensure_ascii=False, sort_keys=True))
if rows <= 0 or cands <= 0:
    raise SystemExit("[ERROR] Empty source candidate allowlist")
PY

echo "[STAGE P1] Merge wiki100k with de old-new_v3 source candidates for source exact matching"
"${SPACY_PY}" - "${BASE_SOURCE_GLOSSARY}" "${CAND_JSONL}" "${MERGED_SOURCE_GLOSSARY}" <<'PY'
import json
import re
import sys
from collections import Counter
from pathlib import Path

base_path = Path(sys.argv[1])
cand_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])

word_re = re.compile(r"[A-Za-z0-9]+(?:[’'][A-Za-z0-9]+)?")
exclude = {
    "a", "an", "the", "this", "that", "these", "those", "his", "her", "hers",
    "him", "he", "she", "it", "its", "they", "them", "their", "theirs",
    "you", "your", "yours", "we", "our", "ours", "i", "me", "my", "mine",
    "what", "which", "who", "whom", "whose", "someone", "somebody", "something",
    "anyone", "anybody", "anything", "everyone", "everybody", "everything",
    "there", "here", "where", "when", "why", "how", "all", "any", "some",
    "one", "two", "both",
}

def term_key(text: str) -> str:
    return " ".join(t.lower() for t in word_re.findall(text or ""))

def keep_candidate(text: str) -> bool:
    toks = term_key(text).split()
    if not toks:
        return False
    if len(toks) > 6:
        return False
    if any(t in exclude for t in toks):
        return False
    if len("".join(toks)) < 3:
        return False
    return True

base = json.loads(base_path.read_text(encoding="utf-8"))
if not isinstance(base, list):
    raise SystemExit(f"[ERROR] base source glossary must be a list: {base_path}")

merged = {}
stats = Counter()
for item in base:
    if not isinstance(item, dict):
        continue
    term = str(item.get("term") or "").strip()
    key = term_key(term)
    if not key:
        continue
    row = dict(item)
    row["term_key"] = key
    merged[key] = row
    stats["base_terms"] += 1

candidate_counts = Counter()
for line in cand_path.open("r", encoding="utf-8"):
    if not line.strip():
        continue
    obj = json.loads(line)
    for cand in obj.get("ner_candidates") or []:
        cand = str(cand or "").strip()
        if not keep_candidate(cand):
            stats["candidate_skipped"] += 1
            continue
        key = term_key(cand)
        candidate_counts[key] += 1
        if key not in merged:
            merged[key] = {
                "term": cand,
                "term_key": key,
                "source": "gigaspeech_oldnewv3_candidate",
                "train_gt_count": 1,
            }
            stats["candidate_added"] += 1
        else:
            old = merged[key]
            old["source"] = f"{old.get('source', 'unknown')}+gigaspeech_oldnewv3_candidate"
            old["train_gt_count"] = int(old.get("train_gt_count") or 0) + 1
            stats["candidate_overrode_existing"] += 1

for key, count in candidate_counts.items():
    merged[key]["train_gt_count"] = max(int(merged[key].get("train_gt_count") or 0), count)

payload = sorted(merged.values(), key=lambda x: str(x.get("term_key") or ""))
out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
stats["merged_terms"] = len(payload)
stats["unique_candidates_kept"] = len(candidate_counts)
print(json.dumps({"merged_source_glossary": out_path.as_posix(), **stats}, ensure_ascii=False, sort_keys=True))
if stats["candidate_added"] <= 0:
    raise SystemExit("[ERROR] No source candidates were added to merged source glossary")
PY

echo "[STAGE A-D] Run MFA/OpenAI rewrite + old-new_v3 term_map builder with source candidate allowlist"
export ROOT_DIR_OVERRIDE="${ROOT_DIR}"
export DATA_ROOT_OVERRIDE="${DATA_ROOT}"
export OUT_ROOT_OVERRIDE="${OUT_ROOT}"
export OUT_NAME_PREFIX_OVERRIDE="${OUT_NAME_PREFIX}"
export LANGS_OVERRIDE="de"
export GPU_DEVICES_CSV_OVERRIDE="${GPU_DEVICES_CSV}"
export NUM_SHARDS_OVERRIDE="${NUM_SHARDS_OVERRIDE:-2}"
export STAGE0_SHARDS_OVERRIDE="${STAGE0_SHARDS_OVERRIDE:-4}"
export SOURCE_GLOSSARY_OVERRIDE="${MERGED_SOURCE_GLOSSARY}"
export SOURCE_CANDIDATE_JSONL_OVERRIDE="${CAND_JSONL}"
export PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP_OVERRIDE="${PREFILTER_REFERENCE_SPANS_FROM_INPUT_TERM_MAP_OVERRIDE:-1}"
export USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI_OVERRIDE="${USE_LEGACY_TERMMAP_SPAN_AS_TARGET_WITHOUT_OPENAI_OVERRIDE:-0}"
export OPENAI_REWRITE_BATCH_OVERRIDE="${OPENAI_REWRITE_BATCH_OVERRIDE:-16}"
export OPENAI_TRANSLATE_BATCH_OVERRIDE="${OPENAI_TRANSLATE_BATCH_OVERRIDE:-64}"
export OPENAI_TRANSLATE_WORKERS_OVERRIDE="${OPENAI_TRANSLATE_WORKERS_OVERRIDE:-8}"
export BATCH_ACROSS_CONVERSATIONS_OVERRIDE="${BATCH_ACROSS_CONVERSATIONS_OVERRIDE:-1}"
export AUDIO_ENCODE_BATCH_OVERRIDE="${AUDIO_ENCODE_BATCH_OVERRIDE:-96}"
export MAX_BATCH_SECONDS_OVERRIDE="${MAX_BATCH_SECONDS_OVERRIDE:-180}"

bash "${BASE_BUILDER}"

echo "[OK] de npfilter New V9 data: ${OUT_DIR}"
