#!/usr/bin/env bash
# Build de New V10 data by preserving retrieved term_map on no-GT chunks.
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

SOURCE_DIR="${SOURCE_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v9_mfa_npfilter_srcunion_lexexact_oldnewv3_de_20260524}"
STAGE2_INPUT="${STAGE2_INPUT_OVERRIDE:-${SOURCE_DIR}/stage2_train_s_de_oldnewv3_mfa_openai_termmap_gtbackfill.jsonl}"
STAGE2_STATS="${STAGE2_STATS_OVERRIDE:-${SOURCE_DIR}/stage3_no_gt_zero_stats.json}"
OUT_DIR="${OUT_DIR_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/speech_llm_new_v10_no_gt_retrieved_termmap_mfa_npfilter_oldnewv3_de_20260525}"
LOG_DIR="${LOG_DIR_OVERRIDE:-${OUT_DIR}/logs}"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

FINAL_TRAIN="${OUT_DIR}/train_s_de_new_v10_no_gt_retrieved_termmap_mfa_npfilter_oldnewv3.jsonl"
FINAL_DEV="${OUT_DIR}/dev_s_de_new_v10_no_gt_retrieved_termmap_mfa_npfilter_oldnewv3_first${DEV_ROWS}.jsonl"
TAG_STATS="${OUT_DIR}/stage3direct_assistant_termtag_stats.json"
TAG_SAMPLES="${OUT_DIR}/stage3direct_assistant_termtag_samples.json"
SUMMARY_JSON="${OUT_DIR}/new_v10_no_gt_retrieved_termmap_mfa_npfilter_oldnewv3_de_summary.json"
VALIDATION_SAMPLES="${OUT_DIR}/new_v10_no_gt_retrieved_no_gt_chunk_samples.json"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-a,an,the,this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything,there,here,where,when,why,how,all,any,some,one,two}"

for p in "${STAGE2_INPUT}" "${WRAP_SCRIPT}"; do
  if [[ ! -s "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_DIR}" "${LOG_DIR}"
if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
  rm -f "${FINAL_TRAIN}" "${FINAL_DEV}" "${TAG_STATS}" "${TAG_SAMPLES}" \
    "${SUMMARY_JSON}" "${VALIDATION_SAMPLES}"
else
  for p in "${FINAL_TRAIN}" "${FINAL_DEV}" "${TAG_STATS}" "${TAG_SAMPLES}" \
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
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] FINAL_TRAIN=${FINAL_TRAIN}"
echo "[INFO] FINAL_DEV=${FINAL_DEV}"
echo "[INFO] DEV_ROWS=${DEV_ROWS}"

echo "[STAGE V10] assistant <term> tags directly from stage2_gtbackfill"
"${PYTHON_BIN}" "${WRAP_SCRIPT}" \
  --input-jsonl "${STAGE2_INPUT}" \
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

echo "[VALIDATE] V10 no-GT retrieved term_map stats"
"${PYTHON_BIN}" - "${STAGE2_INPUT}" "${STAGE2_STATS}" "${TAG_STATS}" "${FINAL_TRAIN}" "${SUMMARY_JSON}" "${VALIDATION_SAMPLES}" <<'PY'
import json
import re
import sys
from pathlib import Path

stage2_input, zero_stats_path, tag_stats_path, final_train, summary_path, samples_path = map(Path, sys.argv[1:])

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

def immediate_assistant(messages, user_idx: int) -> str:
    for msg in messages[user_idx + 1:]:
        if msg.get("role") == "user" and str(msg.get("content") or "").startswith("<audio>"):
            return ""
        if msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""

rows = 0
chunks = 0
gt_chunks = 0
no_gt_chunks = 0
entries = 0
gt_entries = 0
no_gt_entries = 0
nonempty = 0
gt_nonempty = 0
no_gt_nonempty = 0
gt_terms = 0
gt_in_map = 0
malformed = 0
latin_cut = 0
no_gt_immediate_assistant_tagged = 0
samples = []

with final_train.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rows += 1
        obj = json.loads(line)
        messages = obj.get("messages") or []
        gt_by_chunk = obj.get("gt_terms_by_chunk")
        if not isinstance(gt_by_chunk, list):
            raise SystemExit(f"[ERROR] row {rows}: missing gt_terms_by_chunk")
        user_indices = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
        ]
        if len(user_indices) != len(gt_by_chunk):
            raise SystemExit(
                f"[ERROR] row {rows}: audio user count {len(user_indices)} != gt chunks {len(gt_by_chunk)}"
            )
        for m in messages:
            if m.get("role") == "assistant":
                text = str(m.get("content") or "")
                malformed += int(text.count("<term>") != text.count("</term>"))
                latin_cut += int(tag_cuts_latin_word(text))
        for chunk_i, user_idx in enumerate(user_indices):
            content = str(messages[user_idx].get("content") or "")
            lines = term_map_lines(content)
            gt_terms_chunk = gt_by_chunk[chunk_i] or []
            has_gt = bool(gt_terms_chunk)
            chunks += 1
            entries += len(lines)
            nonempty += int(bool(lines))
            term_keys = {line.split("=", 1)[0].strip().casefold() for line in lines}
            if has_gt:
                gt_chunks += 1
                gt_entries += len(lines)
                gt_nonempty += int(bool(lines))
                gt_terms += len(gt_terms_chunk)
                for gt in gt_terms_chunk:
                    if str(gt.get("term") or "").strip().casefold() in term_keys:
                        gt_in_map += 1
            else:
                no_gt_chunks += 1
                no_gt_entries += len(lines)
                no_gt_nonempty += int(bool(lines))
                immediate = immediate_assistant(messages, user_idx)
                no_gt_immediate_assistant_tagged += int("<term>" in immediate or "</term>" in immediate)
                if len(samples) < 20 and lines:
                    samples.append({
                        "row": rows,
                        "chunk": chunk_i,
                        "term_map_preview": lines[:12],
                        "immediate_assistant_has_term_tag": "<term>" in immediate or "</term>" in immediate,
                        "immediate_assistant_preview": immediate[:300],
                    })

if malformed or latin_cut:
    raise SystemExit(f"[ERROR] bad assistant tags: malformed={malformed} latin_cut={latin_cut}")
if no_gt_chunks <= 0:
    raise SystemExit("[ERROR] no no-GT chunks found")
if no_gt_nonempty <= 0:
    raise SystemExit("[ERROR] no-GT chunks have no retrieved term_map entries")

tag_stats = json.loads(tag_stats_path.read_text(encoding="utf-8"))
zero_stats = json.loads(zero_stats_path.read_text(encoding="utf-8")) if zero_stats_path.exists() else {}
summary = {
    "event": "new_v10_no_gt_retrieved_termmap_mfa_npfilter_oldnewv3_de",
    "stage2_input": str(stage2_input),
    "final_train": str(final_train),
    "assistant_tag": tag_stats,
    "new_v9_zero_stats_reference": zero_stats,
    "final_validation": {
        "rows": rows,
        "chunks": chunks,
        "gt_chunks": gt_chunks,
        "no_gt_chunks": no_gt_chunks,
        "entries": entries,
        "gt_entries": gt_entries,
        "no_gt_entries": no_gt_entries,
        "nonempty_chunks": nonempty,
        "gt_nonempty_chunks": gt_nonempty,
        "no_gt_nonempty_chunks": no_gt_nonempty,
        "no_gt_nonempty_rate": no_gt_nonempty / no_gt_chunks if no_gt_chunks else 0.0,
        "avg_no_gt_entries": no_gt_entries / no_gt_chunks if no_gt_chunks else 0.0,
        "gt_terms": gt_terms,
        "gt_in_term_map_rate": gt_in_map / gt_terms if gt_terms else 0.0,
        "malformed_tag_assistant_messages": malformed,
        "latin_word_cut_tag_messages": latin_cut,
        "no_gt_immediate_assistant_tagged": no_gt_immediate_assistant_tagged,
        "no_gt_immediate_assistant_tagged_rate": no_gt_immediate_assistant_tagged / no_gt_chunks if no_gt_chunks else 0.0,
    },
    "manual_no_gt_samples": samples,
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary["final_validation"], ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[OK] ${FINAL_TRAIN}"
echo "[OK] ${FINAL_DEV}"
echo "[OK] ${SUMMARY_JSON}"
