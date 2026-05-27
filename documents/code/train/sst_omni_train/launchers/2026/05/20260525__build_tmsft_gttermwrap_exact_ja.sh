#!/usr/bin/env bash
# Build Japanese TM-SFT data with exact assistant <term> wrapping.
set -euo pipefail

if [[ -d /mnt/taurus/home/jiaxuanluo/InfiniSST ]]; then
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/mnt/taurus/home/jiaxuanluo/InfiniSST}"
else
  ROOT_DIR="${ROOT_DIR_OVERRIDE:-/home/jiaxuanluo/InfiniSST}"
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LANG_CODE="ja"
DATA_ROOT="${DATA_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo}"
SOURCE_JSONL="${SOURCE_JSONL_OVERRIDE:-${DATA_ROOT}/train_s_ja_v4_ner_baseline_aligned_rate1.0_k20_final.jsonl}"
OUT_DIR="${OUT_DIR_OVERRIDE:-${DATA_ROOT}/speech_llm_tmsft_gttermwrap_exact_ja_20260525}"
DEV_ROWS="${DEV_ROWS_OVERRIDE:-355}"
FORCE_OVERWRITE="${FORCE_OVERWRITE:-0}"

DERIVE_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/derive_gt_terms_from_termmap_matches.py"
WRAP_SCRIPT="${ROOT_DIR}/documents/code/train/sst_omni_train/src/wrap_assistant_term_targets.py"

STAGE0_TRAIN="${OUT_DIR}/stage0_train_s_ja_tmsft_exact_gt_from_termmap.jsonl"
STAGE0_DEV="${OUT_DIR}/stage0_dev_s_ja_tmsft_exact_gt_from_termmap_first${DEV_ROWS}.jsonl"
FINAL_TRAIN="${OUT_DIR}/train_s_ja_tmsft_gttermwrap_exact.jsonl"
FINAL_DEV="${OUT_DIR}/dev_s_ja_tmsft_gttermwrap_exact_first${DEV_ROWS}.jsonl"
SUMMARY_JSON="${OUT_DIR}/tmsft_gttermwrap_exact_ja_summary.json"

EXCLUDE_SOURCE_TOKENS="${EXCLUDE_SOURCE_TOKENS_OVERRIDE:-this,that,these,those,his,her,hers,him,he,she,it,its,they,them,their,theirs,you,your,yours,we,our,ours,i,me,my,mine,myself,yourself,himself,herself,itself,ourselves,yourselves,themselves,what,which,who,whom,whose,someone,somebody,something,anyone,anybody,anything,everyone,everybody,everything}"

for p in "${SOURCE_JSONL}" "${DERIVE_SCRIPT}" "${WRAP_SCRIPT}"; do
  if [[ ! -e "${p}" ]]; then
    echo "[ERROR] Missing required path: ${p}" >&2
    exit 3
  fi
done

mkdir -p "${OUT_DIR}"
outputs=(
  "${STAGE0_TRAIN}"
  "${STAGE0_DEV}"
  "${FINAL_TRAIN}"
  "${FINAL_DEV}"
  "${SUMMARY_JSON}"
  "${OUT_DIR}/stage0_train_exact_gt_stats.json"
  "${OUT_DIR}/stage0_train_exact_gt_samples.json"
  "${OUT_DIR}/stage0_dev_exact_gt_first${DEV_ROWS}_stats.json"
  "${OUT_DIR}/stage0_dev_exact_gt_first${DEV_ROWS}_samples.json"
  "${OUT_DIR}/train_gttermwrap_exact_stats.json"
  "${OUT_DIR}/train_gttermwrap_exact_samples.json"
  "${OUT_DIR}/dev_gttermwrap_exact_first${DEV_ROWS}_stats.json"
  "${OUT_DIR}/dev_gttermwrap_exact_first${DEV_ROWS}_samples.json"
)
if [[ "${FORCE_OVERWRITE}" == "1" ]]; then
  rm -f "${outputs[@]}"
else
  for p in "${outputs[@]}"; do
    if [[ -e "${p}" ]]; then
      echo "[ERROR] Output exists: ${p}" >&2
      echo "[ERROR] Set FORCE_OVERWRITE=1 only for an intentional rebuild." >&2
      exit 4
    fi
  done
fi

echo "[INFO] ROOT_DIR=${ROOT_DIR}"
echo "[INFO] SOURCE_JSONL=${SOURCE_JSONL}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] DEV_ROWS=${DEV_ROWS}"

derive_common=(
  --lang-code "${LANG_CODE}"
  --min-target-chars 2
  --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
  --max-terms-per-chunk 16
  --sample-count 200
)

echo "[STAGE 0] derive exact GT terms from embedded TM-SFT term_map"
python3 "${DERIVE_SCRIPT}" \
  --input-jsonl "${SOURCE_JSONL}" \
  --output-jsonl "${STAGE0_TRAIN}" \
  --stats-json "${OUT_DIR}/stage0_train_exact_gt_stats.json" \
  --sample-json "${OUT_DIR}/stage0_train_exact_gt_samples.json" \
  "${derive_common[@]}"

python3 "${DERIVE_SCRIPT}" \
  --input-jsonl "${SOURCE_JSONL}" \
  --output-jsonl "${STAGE0_DEV}" \
  --stats-json "${OUT_DIR}/stage0_dev_exact_gt_first${DEV_ROWS}_stats.json" \
  --sample-json "${OUT_DIR}/stage0_dev_exact_gt_first${DEV_ROWS}_samples.json" \
  --max-rows "${DEV_ROWS}" \
  "${derive_common[@]}"

wrap_common=(
  --lang-code "${LANG_CODE}"
  --tag-template '<term>{translation}</term>'
  --min-target-chars 2
  --max-tags-per-row 16
  --missing-gt-policy error
  --exclude-source-tokens "${EXCLUDE_SOURCE_TOKENS}"
  --exact-require-text-boundaries
  --sample-count 200
)

echo "[STAGE 1] exact assistant target <term> wrapping"
python3 "${WRAP_SCRIPT}" \
  --input-jsonl "${STAGE0_TRAIN}" \
  --output-jsonl "${FINAL_TRAIN}" \
  --stats-json "${OUT_DIR}/train_gttermwrap_exact_stats.json" \
  --sample-json "${OUT_DIR}/train_gttermwrap_exact_samples.json" \
  "${wrap_common[@]}"

python3 "${WRAP_SCRIPT}" \
  --input-jsonl "${STAGE0_DEV}" \
  --output-jsonl "${FINAL_DEV}" \
  --stats-json "${OUT_DIR}/dev_gttermwrap_exact_first${DEV_ROWS}_stats.json" \
  --sample-json "${OUT_DIR}/dev_gttermwrap_exact_first${DEV_ROWS}_samples.json" \
  "${wrap_common[@]}"

echo "[STAGE 2] validate final JSONLs"
python3 - "${SOURCE_JSONL}" "${FINAL_TRAIN}" "${FINAL_DEV}" "${SUMMARY_JSON}" "${OUT_DIR}" "${DEV_ROWS}" <<'PY'
import json
import re
import sys
from pathlib import Path

source, train, dev, summary, out_dir, dev_rows_s = sys.argv[1:]
source = Path(source)
train = Path(train)
dev = Path(dev)
summary = Path(summary)
out_dir = Path(out_dir)
dev_rows = int(dev_rows_s)
latin_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")


def load(name):
    p = out_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def is_latin_alnum(ch):
    return bool(ch) and bool(latin_re.fullmatch(ch))


def bad_term_tags(text):
    malformed = text.count("<term>") != text.count("</term>")
    latin_cut = False
    pos = 0
    while True:
        open_pos = text.find("<term>", pos)
        if open_pos < 0:
            break
        close_pos = text.find("</term>", open_pos + len("<term>"))
        if close_pos < 0:
            malformed = True
            break
        inner_start = open_pos + len("<term>")
        inner_end = close_pos
        if inner_start >= inner_end:
            malformed = True
            break
        before = text[open_pos - 1] if open_pos > 0 else ""
        first = text[inner_start]
        last = text[inner_end - 1]
        after_idx = close_pos + len("</term>")
        after = text[after_idx] if after_idx < len(text) else ""
        if is_latin_alnum(before) and is_latin_alnum(first):
            latin_cut = True
        if is_latin_alnum(after) and is_latin_alnum(last):
            latin_cut = True
        pos = after_idx
    return malformed, latin_cut


def validate(path):
    rows = chunks = gt_terms = gt_chunks = tagged_rows = tagged_assistant_messages = 0
    malformed = latin_cut = termmap_chunks = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows += 1
            obj = json.loads(line)
            messages = obj.get("messages")
            audios = obj.get("audios")
            gt = obj.get("gt_terms_by_chunk")
            if not isinstance(messages, list) or not isinstance(audios, list) or not isinstance(gt, list):
                raise SystemExit(f"[ERROR] malformed row {line_no} in {path}")
            user_indices = [
                i for i, m in enumerate(messages)
                if m.get("role") == "user" and str(m.get("content") or "").startswith("<audio>")
            ]
            if len(user_indices) != len(audios) or len(gt) != len(audios):
                raise SystemExit(
                    f"[ERROR] row {line_no}: user/audio/gt mismatch "
                    f"{len(user_indices)}/{len(audios)}/{len(gt)}"
                )
            chunks += len(audios)
            for idx in user_indices:
                content = str(messages[idx].get("content") or "")
                if "term_map:" in content and "term_map:NONE" not in content:
                    termmap_chunks += 1
            for chunk_terms in gt:
                if chunk_terms:
                    gt_chunks += 1
                gt_terms += len(chunk_terms)
            row_has_tag = False
            for msg in messages:
                if msg.get("role") != "assistant":
                    continue
                text = str(msg.get("content") or "")
                if "<term>" in text:
                    row_has_tag = True
                    tagged_assistant_messages += 1
                bad, cut = bad_term_tags(text)
                malformed += int(bad)
                latin_cut += int(cut)
            tagged_rows += int(row_has_tag)
    if rows <= 0 or chunks <= 0:
        raise SystemExit(f"[ERROR] empty output: {path}")
    if malformed or latin_cut:
        raise SystemExit(f"[ERROR] bad tags in {path}: malformed={malformed} latin_cut={latin_cut}")
    return {
        "rows": rows,
        "chunks": chunks,
        "termmap_chunks": termmap_chunks,
        "gt_chunks": gt_chunks,
        "gt_terms": gt_terms,
        "tagged_rows": tagged_rows,
        "tagged_assistant_messages": tagged_assistant_messages,
        "malformed_tag_assistant_messages": malformed,
        "latin_word_cut_tag_messages": latin_cut,
    }


summary_obj = {
    "event": "tmsft_gttermwrap_exact_ja",
    "source_jsonl": str(source),
    "final_train": str(train),
    "final_dev": str(dev),
    "policy": {
        "gt_derivation": "embedded user term_map target must appear exactly in current/future assistant text",
        "assistant_wrapping": "exact future assistant substring replacement only",
        "local_rewrite": False,
        "fuzzy_gt_derivation": False,
        "exact_require_text_boundaries": True,
        "term_map_unchanged": True,
        "llm_variant_augmentation": False,
        "no_gt_zero": False,
    },
    "stats": {
        "stage0_train_exact_gt": load("stage0_train_exact_gt_stats.json"),
        "stage0_dev_exact_gt": load(f"stage0_dev_exact_gt_first{dev_rows}_stats.json"),
        "train_wrap": load("train_gttermwrap_exact_stats.json"),
        "dev_wrap": load(f"dev_gttermwrap_exact_first{dev_rows}_stats.json"),
        "final_train_validation": validate(train),
        "final_dev_validation": validate(dev),
    },
}
summary.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary_obj["stats"]["final_train_validation"], ensure_ascii=False, sort_keys=True))
print(json.dumps(summary_obj["stats"]["final_dev_validation"], ensure_ascii=False, sort_keys=True))
PY

echo "[DONE] ${FINAL_TRAIN}"
echo "[DONE] ${FINAL_DEV}"
echo "[DONE] ${SUMMARY_JSON}"
