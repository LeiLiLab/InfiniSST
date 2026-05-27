#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/jiaxuanluo/InfiniSST}"
cd "${REPO_ROOT}"

RUN_STAMP="20260524T1748_medicine_rasst_zh_lm1_max80_sharedaudio_batch"
EVENT_ID="20260524T1748__simuleval__medicine_hardraw_rasst_zh_lm1_max80_sharedaudio_batch"
OUTROOT="/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_${RUN_STAMP}"
RUNDIR="${OUTROOT}/zh/dmedhard5_new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_max80_raw_lm1_k10_th0.78_ghard_medicine_glossary_raw_llm_judge_manual_zh215_unique212_ppmedicine5_hardraw"
EVAL_TSV="${RUNDIR}/eval_results.tsv"
INSTANCES_LOG="${RUNDIR}/instances.log"
MANIFEST="documents/code/simuleval/manifests/2026/05/${EVENT_ID}.json"
NOTES="documents/code/simuleval/notes/2026/05/20260524__medicine_hardraw_rasst_zh_lm1_max80_batch.md"
FIG_SCRIPT="documents/code/simuleval/src/build_main_result_tables_and_figures_20260524.py"
FIG_PDF="documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures/medicine_main_result.pdf"
REPORT_TSV="documents/code/simuleval/reports/20260524_main_result_data.tsv"
LOGROOT="/mnt/gemini/data1/jiaxuanluo/logs/medicine_hardraw_hn1024_tau078_new_v9_batch_${RUN_STAMP}"
WATCH_LOG="${LOGROOT}/postprocess_watch.log"

mkdir -p "${LOGROOT}"
exec >>"${WATCH_LOG}" 2>&1

echo "[watch] start $(date -Is)"
echo "[watch] rundir=${RUNDIR}"

deadline=$((SECONDS + ${WATCH_TIMEOUT_SEC:-18000}))
while (( SECONDS < deadline )); do
  line_count=0
  if [[ -f "${INSTANCES_LOG}" ]]; then
    line_count="$(wc -l < "${INSTANCES_LOG}" || echo 0)"
  fi
  if [[ -s "${EVAL_TSV}" && "${line_count}" -ge 5 ]]; then
    echo "[watch] success condition met at $(date -Is), instances=${line_count}"
    break
  fi
  echo "[watch] waiting $(date -Is), instances=${line_count}, eval_tsv=$([[ -s "${EVAL_TSV}" ]] && echo yes || echo no)"
  sleep 120
done

line_count=0
if [[ -f "${INSTANCES_LOG}" ]]; then
  line_count="$(wc -l < "${INSTANCES_LOG}" || echo 0)"
fi

if [[ ! -s "${EVAL_TSV}" || "${line_count}" -lt 5 ]]; then
  echo "[watch] failed or timed out at $(date -Is), instances=${line_count}, eval_tsv=$([[ -s "${EVAL_TSV}" ]] && echo yes || echo no)"
  python3 - <<'PY'
import json
from pathlib import Path
manifest = Path("documents/code/simuleval/manifests/2026/05/20260524T1748__simuleval__medicine_hardraw_rasst_zh_lm1_max80_sharedaudio_batch.json")
data = json.loads(manifest.read_text())
data["status"] = "failed"
meta = data.setdefault("metadata", {})
meta["postprocess_status"] = "failed_or_timeout"
manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
PY
  python3 documents/code/general/experiment_event.py register "${MANIFEST}" || true
  ~/bin/codex-notify --delay 8 --detach --workspace "${REPO_ROOT}" \
    "Codex finished: zh medicine RASST lm1 max80 eval did not complete; see ${WATCH_LOG}" || true
  exit 1
fi

python3 - <<'PY'
import csv
import json
from pathlib import Path

event_id = "20260524T1748__simuleval__medicine_hardraw_rasst_zh_lm1_max80_sharedaudio_batch"
manifest = Path(f"documents/code/simuleval/manifests/2026/05/{event_id}.json")
notes = Path("documents/code/simuleval/notes/2026/05/20260524__medicine_hardraw_rasst_zh_lm1_max80_batch.md")
eval_tsv = Path("/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T1748_medicine_rasst_zh_lm1_max80_sharedaudio_batch/zh/dmedhard5_new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_max80_raw_lm1_k10_th0.78_ghard_medicine_glossary_raw_llm_judge_manual_zh215_unique212_ppmedicine5_hardraw/eval_results.tsv")

with eval_tsv.open() as f:
    row = next(csv.DictReader(f, delimiter="\t"))

metrics = {
    "BLEU": float(row["BLEU"]),
    "StreamLAAL": float(row["StreamLAAL"]),
    "TERM_ACC": float(row["TERM_ACC"]),
    "REAL_TERM_ADOPT": float(row["REAL_TERM_ADOPT"]),
    "TERM_FCR": float(row["TERM_FCR"]),
    "TERM_CORRECT": int(row["TERM_CORRECT"]),
    "TERM_TOTAL": int(row["TERM_TOTAL"]),
}

data = json.loads(manifest.read_text())
data["status"] = "success"
data["artifacts"] = [
    a for a in data.get("artifacts", [])
    if a.get("role") not in {"eval_results", "eval_output"}
]
data.setdefault("artifacts", []).extend([
    {
        "role": "eval_output",
        "type": "directory",
        "direction": "output",
        "path": str(eval_tsv.parent),
        "metadata": {"lang": "zh", "lm": 1, "max_new_tokens": 80},
    },
    {
        "role": "eval_results",
        "type": "tsv",
        "direction": "output",
        "path": str(eval_tsv),
        "metadata": metrics,
    },
])
meta = data.setdefault("metadata", {})
meta["postprocess_status"] = "success"
meta["metrics"] = metrics
manifest.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

verdict = f"""## Verdict

SUCCESS. The five-sample zh medicine hardraw RASST lm=1 batch completed with `max_new_tokens=80`.

- eval results: `{eval_tsv}`
- BLEU: {metrics['BLEU']:.4f}
- StreamLAAL: {metrics['StreamLAAL']:.2f}
- TERM_ACC: {metrics['TERM_ACC']:.4f} ({metrics['TERM_CORRECT']}/{metrics['TERM_TOTAL']})
- REAL_TERM_ADOPT: {metrics['REAL_TERM_ADOPT']:.4f}
- TERM_FCR: {metrics['TERM_FCR']:.4f}
"""

text = notes.read_text()
head = text.split("## Verdict", 1)[0].rstrip()
notes.write_text(head + "\n\n" + verdict + "\n")
PY

python3 documents/code/general/experiment_event.py register "${MANIFEST}" || true
python3 "${FIG_SCRIPT}"

echo "[watch] updated figure ${FIG_PDF}"
grep -n $'medicine_hardraw\tRASST\tzh\t1' "${REPORT_TSV}" || true
~/bin/codex-notify --delay 8 --detach --workspace "${REPO_ROOT}" \
  "Codex finished: zh medicine RASST lm1 max80 eval done; medicine_main_result.pdf updated" || true
echo "[watch] done $(date -Is)"
