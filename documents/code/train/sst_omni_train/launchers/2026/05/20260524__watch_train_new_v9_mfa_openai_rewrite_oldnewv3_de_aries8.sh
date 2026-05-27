#!/usr/bin/env bash
# Wait for clean de New V9 data, validate it, then start 8-GPU SFT on aries.
set -euo pipefail

ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/logs}"
TRAIN_LAUNCHER="${ROOT_DIR}/documents/code/train/sst_omni_train/launchers/2026/05/20260524__speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_ja_taurus4_r32a64_tp2.sh"
POLL_SEC="${POLL_SEC_OVERRIDE:-45}"
LANG_CODE="de"
mkdir -p "${LOG_ROOT}"

validate_or_write_summary() {
  local lang="${LANG_CODE}"
  local data_dir="${DATA_ROOT}/speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_${lang}_20260524"
  local train="${data_dir}/train_s_${lang}_new_v9_mfa_openai_rewrite_oldnewv3.jsonl"
  local dev="${data_dir}/dev_s_${lang}_new_v9_mfa_openai_rewrite_oldnewv3_first355.jsonl"
  local summary="${data_dir}/new_v9_mfa_openai_rewrite_oldnewv3_${lang}_summary.json"
  [[ -s "${train}" && -s "${dev}" ]] || return 1
  python3 - "${lang}" "${train}" "${summary}" <<'PY'
import json
import re
import sys
from pathlib import Path

lang, final_train, summary = sys.argv[1:]
final_train = Path(final_train)
summary = Path(summary)
base = final_train.parent
source_jsonl = f"/mnt/gemini/data1/jiaxuanluo/train_s_{lang}_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl"

def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}

latin_alnum_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")

def is_latin_alnum(ch):
    return bool(ch) and bool(latin_alnum_re.fullmatch(ch))

def tag_cuts_latin_word(text):
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

rows = malformed = latin_cut = gt_terms = term_map_terms = gt_in_map = 0
no_gt_chunks = no_gt_zero_chunks = 0
sample_rows = []
with final_train.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rows += 1
        obj = json.loads(line)
        messages = obj.get("messages") or []
        gt_by_chunk = obj.get("gt_terms_by_chunk") or []
        audio_user_idx = [
            i for i, m in enumerate(messages)
            if m.get("role") == "user" and "<audio>" in str(m.get("content") or "")
        ]
        if len(audio_user_idx) != len(obj.get("audios") or []):
            raise SystemExit(f"[ERROR] audio/user count mismatch row={rows}")
        for m in messages:
            if m.get("role") != "assistant":
                continue
            text = str(m.get("content") or "")
            if text.count("<term>") != text.count("</term>"):
                malformed += 1
            if tag_cuts_latin_word(text):
                latin_cut += 1
        for chunk_i, gt_terms_chunk in enumerate(gt_by_chunk):
            gt_terms += len(gt_terms_chunk)
            user_msg = messages[audio_user_idx[chunk_i]]
            content = str(user_msg.get("content") or "")
            term_lines = [x for x in content.splitlines() if "=" in x]
            term_map_terms += len(term_lines)
            if not gt_terms_chunk:
                no_gt_chunks += 1
                if "term_map:NONE" in content:
                    no_gt_zero_chunks += 1
            term_keys = {str(x.split("=", 1)[0]).strip().casefold() for x in term_lines}
            for gt in gt_terms_chunk:
                if str(gt.get("term") or "").strip().casefold() in term_keys:
                    gt_in_map += 1
        if len(sample_rows) < 20:
            sample_rows.append({
                "utter_id": obj.get("utter_id"),
                "source_chunk_mfa_text_by_chunk": obj.get("source_chunk_mfa_text_by_chunk", [])[:4],
                "gt_terms_by_chunk": gt_by_chunk[:4],
                "first_user_term_map": next(
                    (str(messages[i].get("content") or "") for i in audio_user_idx
                     if "term_map:" in str(messages[i].get("content") or "")),
                    "",
                )[:800],
                "assistant_tags": [
                    str(m.get("content") or "") for m in messages
                    if m.get("role") == "assistant" and "<term>" in str(m.get("content") or "")
                ][:3],
            })

if rows <= 0 or gt_terms <= 0:
    raise SystemExit(f"[ERROR] invalid counts rows={rows} gt_terms={gt_terms}")
if malformed or latin_cut:
    raise SystemExit(f"[ERROR] bad tags malformed={malformed} latin_cut={latin_cut}")

summary_obj = {
    "event": f"new_v9_mfa_openai_rewrite_oldnewv3_{lang}",
    "source_jsonl": source_jsonl,
    "final_train": str(final_train),
    "stage0": load(base / "stage0_mfa_openai_rewrite_gt_stats.json"),
    "retriever_translation": load(base / "stage1_retriever_translation_stats.json"),
    "no_gt_zero": load(base / "stage3_no_gt_zero_stats.json"),
    "assistant_tag": load(base / "stage4_assistant_termtag_stats.json"),
    "final_validation": {
        "rows": rows,
        "gt_terms": gt_terms,
        "term_map_terms": term_map_terms,
        "gt_in_term_map_rate": gt_in_map / gt_terms if gt_terms else 0.0,
        "no_gt_zero_rate": no_gt_zero_chunks / no_gt_chunks if no_gt_chunks else 0.0,
        "malformed_tag_assistant_messages": malformed,
        "latin_word_cut_tag_messages": latin_cut,
    },
    "manual_inspection_samples": sample_rows,
}
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary_obj["final_validation"], ensure_ascii=False, sort_keys=True))
PY
}

start_train() {
  local stamp out err pidfile
  stamp="$(date -u +%Y%m%dT%H%M%S)"
  out="${LOG_ROOT}/train_speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_aries8_${stamp}.out"
  err="${LOG_ROOT}/train_speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_aries8_${stamp}.err"
  pidfile="${LOG_ROOT}/train_speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_aries8_${stamp}.pid"
  setsid bash -lc "
    set -euo pipefail
    cd '${ROOT_DIR}'
    LANG_CODE_OVERRIDE=de \
    ROOT_DIR_OVERRIDE='${ROOT_DIR}' \
    HOST_GPU_DEVICES_OVERRIDE=0,1,2,3,4,5,6,7 \
    EXPECTED_GPUS_OVERRIDE=8 \
    NPROC_PER_NODE_OVERRIDE=8 \
    GLOBAL_BATCH_SIZE_OVERRIDE=8 \
    COMPUTE_TAG_OVERRIDE=aries8 \
    SAVE_BASE_OVERRIDE='${DATA_ROOT}/slm/speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_r32a64_tp2_aries8' \
    TRAIN_LOG_DIR_OVERRIDE='${LOG_ROOT}/speech_llm_new_v9_mfa_openai_rewrite_oldnewv3_de_r32a64_tp2_aries8' \
    WANDB_EXP_PREFIX_OVERRIDE=speech-llm-new_v9-mfa-openai-oldnewv3-de-r32a64-tp2-aries8 \
    bash '${TRAIN_LAUNCHER}'
  " > "${out}" 2> "${err}" < /dev/null &
  echo "$!" > "${pidfile}"
  echo "[STARTED] de aries8 pid=$(cat "${pidfile}") out=${out} err=${err}"
}

echo "[INFO] de aries8 watcher started at $(date -u --iso-8601=seconds)"
while true; do
  if validate_or_write_summary; then
    start_train
    exit 0
  fi
  echo "[WAIT] de data not validated at $(date -u --iso-8601=seconds)"
  sleep "${POLL_SEC}"
done
