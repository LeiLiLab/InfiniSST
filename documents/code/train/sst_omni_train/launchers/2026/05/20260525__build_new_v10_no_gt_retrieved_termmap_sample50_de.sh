#!/usr/bin/env bash
# Build de New V10 sample50 data by downsampling retrieved term_map on no-GT chunks.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

CONDA_PREFIX_OVERRIDE="${CONDA_PREFIX_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv}"
if [[ -d "${CONDA_PREFIX_OVERRIDE}" ]]; then
  export PATH="${CONDA_PREFIX_OVERRIDE}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX_OVERRIDE}/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

PYTHON_BIN="${PYTHON_BIN_OVERRIDE:-python3}"
LANG_CODE="de"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"
NO_GT_KEEP_PROB="${NO_GT_KEEP_PROB_OVERRIDE:-0.5}"
NO_GT_MAX_TERMS="${NO_GT_MAX_TERMS_OVERRIDE:-0}"
SAMPLE_UNIT="${SAMPLE_UNIT_OVERRIDE:-term}"
SAMPLE_SEED="${SAMPLE_SEED_OVERRIDE:-20260525_new_v10_de_sample50}"

SOURCE_DIR="${SOURCE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524}"
STAGE2_INPUT="${STAGE2_INPUT_OVERRIDE:-${SOURCE_DIR}/stage2_train_s_de_oldnewv3_mfa_openai_termmap_gtbackfill.jsonl}"
BASELINE_TMSFT_JSONL="${BASELINE_TMSFT_JSONL_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/train_s_de_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_de_20260525}"
LOG_DIR="${LOG_DIR_OVERRIDE:-${OUT_DIR}/logs}"

SAMPLE_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/sample_no_gt_termmap_chunks.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

SAMPLED_STAGE2="${OUT_DIR}/stage2_train_s_de_oldnewv3_mfa_openai_termmap_gtbackfill_no_gt_sample50.jsonl"
SAMPLE_STATS="${OUT_DIR}/stage2_no_gt_sample50_stats.json"
SAMPLE_SAMPLES="${OUT_DIR}/stage2_no_gt_sample50_samples.json"
FINAL_TRAIN="${OUT_DIR}/train_s_de_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3.jsonl"
FINAL_DEV="${OUT_DIR}/dev_s_de_new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_first${DEV_ROWS}.jsonl"
TAG_STATS="${OUT_DIR}/stage3direct_assistant_termtag_stats.json"
TAG_SAMPLES="${OUT_DIR}/stage3direct_assistant_termtag_samples.json"
SUMMARY_JSON="${OUT_DIR}/new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_de_summary.json"
VALIDATION_SAMPLES="${OUT_DIR}/new_v10_no_gt_retrieved_sample50_no_gt_chunk_samples.json"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-a,an,the,this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything,there,here,where,when,why,how,all,any,some,one,two}"

for p in "${STAGE2_INPUT}" "${BASELINE_TMSFT_JSONL}" "${SAMPLE_SCRIPT}" "${WRAP_SCRIPT}"; do
  if [[ ! -s "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
  rm -f "${SAMPLED_STAGE2}" "${SAMPLE_STATS}" "${SAMPLE_SAMPLES}" \
    "${FINAL_TRAIN}" "${FINAL_DEV}" "${TAG_STATS}" "${TAG_SAMPLES}" \
    "${SUMMARY_JSON}" "${VALIDATION_SAMPLES}"
else
  for p in "${SAMPLED_STAGE2}" "${SAMPLE_STATS}" "${SAMPLE_SAMPLES}" \
    "${FINAL_TRAIN}" "${FINAL_DEV}" "${TAG_STATS}" "${TAG_SAMPLES}" \
    "${SUMMARY_JSON}" "${VALIDATION_SAMPLES}"; do
    if [[ -e "${p}" ]]; then
      echo "[ERROR] Output exists: ${p}" >&2
      echo "[ERROR] Set FORCE_OVERWRITE=1 only for intentional reruns." >&2
      exit 4
    fi
  done
fi

cd "${ROOT_DIR}"
echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] STAGE2_INPUT=${STAGE2_INPUT}"
echo "[INFO] BASELINE_TMSFT_JSONL=${BASELINE_TMSFT_JSONL}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NO_GT_KEEP_PROB=${NO_GT_KEEP_PROB}"
echo "[INFO] NO_GT_MAX_TERMS=${NO_GT_MAX_TERMS}"
echo "[INFO] SAMPLE_UNIT=${SAMPLE_UNIT}"

echo "[STAGE SAMPLE] no-GT retrieved term_map sample50"
"${PYTHON_BIN}" "${SAMPLE_SCRIPT}" \
  --input-jsonl "${STAGE2_INPUT}" \
  --output-jsonl "${SAMPLED_STAGE2}" \
  --stats-json "${SAMPLE_STATS}" \
  --sample-json "${SAMPLE_SAMPLES}" \
  --keep-prob "${NO_GT_KEEP_PROB}" \
  --max-no-gt-terms "${NO_GT_MAX_TERMS}" \
  --sample-unit "${SAMPLE_UNIT}" \
  --seed "${SAMPLE_SEED}" \
  --max-samples 80 2>&1 | tee "${LOG_DIR}/stage2_no_gt_sample50_de.log"

echo "[STAGE WRAP] assistant <term> tags from sampled stage2"
"${PYTHON_BIN}" "${WRAP_SCRIPT}" \
  --input-jsonl "${SAMPLED_STAGE2}" \
  --output-jsonl "${FINAL_TRAIN}" \
  --stats-json "${TAG_STATS}" \
  --sample-json "${TAG_SAMPLES}" \
  --lang-code "${LANG_CODE}" \
  --tag-template '<term>{translation}</term>' \
  --min-target-chars 2 \
  --max-tags-per-row 16 \
  --missing-gt-policy error \
  --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}" \
  --exact-require-text-boundaries \
  --enable-local-rewrite \
  --rewrite-boundary-only \
  --rewrite-delay-boundary-prefix \
  --rewrite-delay-boundary-min-prefix-chars 2 \
  --rewrite-require-text-boundaries \
  --sample-count 200 2>&1 | tee "${LOG_DIR}/stage3direct_wrap_assistant_terms_de.log"

echo "[DEV] first ${DEV_ROWS} rows"
head -n "${DEV_ROWS}" "${FINAL_TRAIN}" > "${FINAL_DEV}"

echo "[VALIDATE] sample50 density and tag checks"
"${PYTHON_BIN}" - "${BASELINE_TMSFT_JSONL}" "${SAMPLE_STATS}" "${TAG_STATS}" "${FINAL_TRAIN}" "${SUMMARY_JSON}" "${VALIDATION_SAMPLES}" <<'PY'
import json
import re
import sys
from pathlib import Path

baseline_path, sample_stats_path, tag_stats_path, final_train, summary_path, samples_path = map(Path, sys.argv[1:])
latin_alnum_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def is_latin_alnum(ch: str) -> bool:
    return bool(ch) and bool(latin_alnum_re.fullmatch(ch))

def tag_cuts_latin_word(text: str) -> bool:
    start = 0
    while True:
        open_pos = text.find("<term>", start)
        if open_pos < 0:
            return False
        close_pos = text.find("</term>", open_pos + len("<term>"))
        if close_pos < 0:
            return True
        inner_start = open_pos + len("<term>")
        inner_end = close_pos
        if inner_start >= inner_end:
            return True
        before = text[open_pos - 1] if open_pos > 0 else ""
        first = text[inner_start]
        last = text[inner_end - 1]
        after_idx = close_pos + len("</term>")
        after = text[after_idx] if after_idx < len(text) else ""
        if is_latin_alnum(before) and is_latin_alnum(first):
            return True
        if is_latin_alnum(after) and is_latin_alnum(last):
            return True
        start = after_idx

def term_map_lines(content: str):
    if "term_map:NONE" in content:
        return []
    marker = "term_map:"
    idx = content.find(marker)
    if idx < 0:
        return []
    return [line.strip() for line in content[idx + len(marker):].splitlines() if "=" in line]

def user_messages(obj):
    return [
        m for m in (obj.get("messages") or [])
        if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
    ]

def assistant_after(messages, user_idx: int) -> str:
    for msg in messages[user_idx + 1:]:
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("<audio>"):
            return ""
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""

def row_key(obj):
    if obj.get("utter_id"):
        return str(obj["utter_id"])
    audios = obj.get("audios") or []
    if not audios:
        return ""
    parts = Path(audios[0]).parts
    return f"{parts[-3]}_{parts[-2]}" if len(parts) >= 3 else str(audios[0])

baseline = {}
with baseline_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            baseline[row_key(obj)] = obj

stats = {
    "rows": 0,
    "chunks": 0,
    "gt_chunks": 0,
    "no_gt_chunks": 0,
    "entries": 0,
    "gt_entries": 0,
    "no_gt_entries": 0,
    "nonempty_chunks": 0,
    "gt_nonempty_chunks": 0,
    "no_gt_nonempty_chunks": 0,
    "gt_terms": 0,
    "gt_in_term_map": 0,
    "malformed_tag_assistant_messages": 0,
    "latin_word_cut_tag_messages": 0,
    "no_gt_immediate_assistant_tagged": 0,
    "baseline_no_gt_entries": 0,
    "baseline_no_gt_nonempty_chunks": 0,
    "baseline_missing_rows": 0,
}
samples = []

with final_train.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        stats["rows"] += 1
        obj = json.loads(line)
        messages = obj.get("messages") or []
        gt_by_chunk = obj.get("gt_terms_by_chunk")
        if not isinstance(gt_by_chunk, list):
            raise SystemExit(f"[ERROR] row {stats['rows']}: missing gt_terms_by_chunk")
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
        ]
        if len(user_indices) != len(gt_by_chunk):
            raise SystemExit(f"[ERROR] row {stats['rows']}: user chunks != gt chunks")
        base = baseline.get(row_key(obj))
        if base is None:
            stats["baseline_missing_rows"] += 1
            base_users = []
        else:
            base_users = user_messages(base)
            if len(base_users) != len(user_indices):
                raise SystemExit(f"[ERROR] row {stats['rows']}: baseline chunk mismatch")

        for msg in messages:
            if msg.get("role") == "assistant":
                text = str(msg.get("content") or "")
                stats["malformed_tag_assistant_messages"] += int(text.count("<term>") != text.count("</term>"))
                stats["latin_word_cut_tag_messages"] += int(tag_cuts_latin_word(text))

        for chunk_i, user_idx in enumerate(user_indices):
            lines = term_map_lines(str(messages[user_idx].get("content") or ""))
            gt_terms_chunk = gt_by_chunk[chunk_i] or []
            has_gt = bool(gt_terms_chunk)
            stats["chunks"] += 1
            stats["entries"] += len(lines)
            stats["nonempty_chunks"] += int(bool(lines))
            term_keys = {line.split("=", 1)[0].strip().casefold() for line in lines}
            if has_gt:
                stats["gt_chunks"] += 1
                stats["gt_entries"] += len(lines)
                stats["gt_nonempty_chunks"] += int(bool(lines))
                stats["gt_terms"] += len(gt_terms_chunk)
                for gt in gt_terms_chunk:
                    if str(gt.get("term") or "").strip().casefold() in term_keys:
                        stats["gt_in_term_map"] += 1
            else:
                stats["no_gt_chunks"] += 1
                stats["no_gt_entries"] += len(lines)
                stats["no_gt_nonempty_chunks"] += int(bool(lines))
                immediate = assistant_after(messages, user_idx)
                stats["no_gt_immediate_assistant_tagged"] += int("<term>" in immediate or "</term>" in immediate)
                if base_users:
                    base_lines = term_map_lines(str(base_users[chunk_i].get("content") or ""))
                    stats["baseline_no_gt_entries"] += len(base_lines)
                    stats["baseline_no_gt_nonempty_chunks"] += int(bool(base_lines))
                if len(samples) < 20 and lines:
                    samples.append({
                        "row": stats["rows"],
                        "chunk": chunk_i,
                        "term_map_preview": lines[:12],
                        "immediate_assistant_has_term_tag": "<term>" in immediate or "</term>" in immediate,
                        "immediate_assistant_preview": immediate[:300],
                    })

if stats["malformed_tag_assistant_messages"] or stats["latin_word_cut_tag_messages"]:
    raise SystemExit(
        "[ERROR] bad assistant tags: "
        f"malformed={stats['malformed_tag_assistant_messages']} "
        f"latin_cut={stats['latin_word_cut_tag_messages']}"
    )
if stats["baseline_missing_rows"]:
    raise SystemExit(f"[ERROR] missing baseline rows: {stats['baseline_missing_rows']}")
if not stats["no_gt_chunks"]:
    raise SystemExit("[ERROR] no no-GT chunks found")

sample_stats = json.loads(sample_stats_path.read_text(encoding="utf-8"))
tag_stats = json.loads(tag_stats_path.read_text(encoding="utf-8"))
validation = {
    **stats,
    "no_gt_nonempty_rate": stats["no_gt_nonempty_chunks"] / stats["no_gt_chunks"],
    "avg_no_gt_entries": stats["no_gt_entries"] / stats["no_gt_chunks"],
    "baseline_no_gt_nonempty_rate": stats["baseline_no_gt_nonempty_chunks"] / stats["no_gt_chunks"],
    "baseline_avg_no_gt_entries": stats["baseline_no_gt_entries"] / stats["no_gt_chunks"],
    "no_gt_entry_ratio_vs_baseline": stats["no_gt_entries"] / stats["baseline_no_gt_entries"],
    "no_gt_nonempty_ratio_vs_baseline": stats["no_gt_nonempty_chunks"] / stats["baseline_no_gt_nonempty_chunks"],
    "gt_in_term_map_rate": stats["gt_in_term_map"] / stats["gt_terms"],
    "no_gt_immediate_assistant_tagged_rate": stats["no_gt_immediate_assistant_tagged"] / stats["no_gt_chunks"],
}
summary = {
    "event": "new_v10_no_gt_retrieved_termmap_sample50_mfa_npfilter_oldnewv3_de",
    "baseline_tmsft_jsonl": str(baseline_path),
    "final_train": str(final_train),
    "sample_stage": sample_stats,
    "assistant_tag": tag_stats,
    "final_validation": validation,
    "manual_no_gt_samples": samples,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[OK] ${SAMPLED_STAGE2}"
echo "[OK] ${FINAL_TRAIN}"
echo "[OK] ${FINAL_DEV}"
echo "[OK] ${SUMMARY_JSON}"
