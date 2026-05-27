#!/usr/bin/env bash
set -euo pipefail

WRAPPER_PID="${WRAPPER_PID_OVERRIDE:-1171282}"
POSTEVAL_PID="${POSTEVAL_PID_OVERRIDE:-}"
WAIT_SECONDS="${WAIT_SECONDS_OVERRIDE:-60}"
MAX_POLLS="${MAX_POLLS_OVERRIDE:-720}"
SUMMARY_ROOT="${SUMMARY_ROOT_OVERRIDE:-/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_origin_summary}"
OUT_TSV="${OUT_TSV_OVERRIDE:-${SUMMARY_ROOT}.tsv}"
OUT_MD="${OUT_MD_OVERRIDE:-${SUMMARY_ROOT}.md}"

for _ in $(seq 1 "${MAX_POLLS}"); do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ssh aries "ps -p ${WRAPPER_PID} >/dev/null 2>&1"; then
    echo "[${ts}] waiting for aries wrapper pid=${WRAPPER_PID}"
    sleep "${WAIT_SECONDS}"
    continue
  fi
  echo "[${ts}] aries wrapper exited; collecting zh origin summary"
  break
done

if ssh aries "ps -p ${WRAPPER_PID} >/dev/null 2>&1"; then
  echo "[ERROR] timed out waiting for aries wrapper pid=${WRAPPER_PID}" >&2
  exit 2
fi

if [[ -n "${POSTEVAL_PID}" ]]; then
  for _ in $(seq 1 "${MAX_POLLS}"); do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if ps -p "${POSTEVAL_PID}" >/dev/null 2>&1; then
      echo "[${ts}] waiting for local hard-manual post-eval pid=${POSTEVAL_PID}"
      sleep "${WAIT_SECONDS}"
      continue
    fi
    echo "[${ts}] local hard-manual post-eval exited; collecting zh origin summary"
    break
  done

  if ps -p "${POSTEVAL_PID}" >/dev/null 2>&1; then
    echo "[ERROR] timed out waiting for hard-manual post-eval pid=${POSTEVAL_PID}" >&2
    exit 2
  fi
fi

sleep 30
python - "${OUT_TSV}" "${OUT_MD}" <<'PY'
import csv
import json
import sys
from pathlib import Path

out_tsv = Path(sys.argv[1])
out_md = Path(sys.argv[2])

settings = [
    {
        "lm": "1",
        "samples": "404 545006 596001 605000 606",
        "eval_policy": "hard_llm_manual_check",
        "out_base": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm1_aries01"),
        "setting": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm1_aries01/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs0.96_hs0.48_lm1_k210_k110_th0p0"),
        "eval_name": "eval_results_streamlaal_term.hard_llm_manual_check.tsv",
        "miss_name": "term_misses.hard_llm_manual_check.zh_lm1.tsv",
        "summary_name": "term_miss_summary.hard_llm_manual_check.zh_lm1.tsv",
    },
    {
        "lm": "2",
        "samples": "404 545006 596001 605000 606",
        "eval_policy": "hard_llm_manual_check_reposteval_20260522_generation",
        "out_base": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260522"),
        "setting": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260522/zh/gigaspeech-zh-s_origin-bsz4_gmedicine_gt571_abbrev_restored__medicine5_cs1.92_hs0.48_lm2_k210_k110_th0p0"),
        "eval_name": "eval_results_streamlaal_term.hard_llm_manual_check.tsv",
        "miss_name": "term_misses.hard_llm_manual_check.zh_lm2.tsv",
        "summary_name": "term_miss_summary.hard_llm_manual_check.zh_lm2.tsv",
    },
    {
        "lm": "3",
        "samples": "404 545006 596001 605000 606",
        "eval_policy": "hard_llm_manual_check",
        "out_base": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm3_aries67"),
        "setting": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm3_aries67/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs2.88_hs0.48_lm3_k210_k110_th0p0"),
        "eval_name": "eval_results_streamlaal_term.hard_llm_manual_check.tsv",
        "miss_name": "term_misses.hard_llm_manual_check.zh_lm3.tsv",
        "summary_name": "term_miss_summary.hard_llm_manual_check.zh_lm3.tsv",
    },
    {
        "lm": "4",
        "samples": "404 545006 596001 606 605000",
        "eval_policy": "hard_llm_manual_check_full5_aries4_plus_taurus605000",
        "out_base": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm4_with605000_from_taurus_orig80"),
        "setting": Path("/mnt/gemini/data1/jiaxuanluo/medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm4_with605000_from_taurus_orig80/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs3.84_hs0.48_lm4_k210_k110_th0p0"),
        "eval_name": "eval_results_streamlaal_term.hard_llm_manual_check.tsv",
        "miss_name": "term_misses.hard_llm_manual_check.zh_lm4_full5.tsv",
        "summary_name": "term_miss_summary.hard_llm_manual_check.zh_lm4_full5.tsv",
    },
]

fields = [
    "lm", "status", "eval_policy", "samples", "sample_count", "seconds", "minutes",
    "instances", "BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC",
    "TERM_CORRECT", "TERM_TOTAL", "miss_occurrences", "unique_missed_term_translations",
    "eval_tsv", "misses_tsv", "summary_tsv", "output_dir",
]


def read_timing(out_base: Path, lm: str):
    path = out_base / "timing.tsv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("lm") == lm:
                return row
    return {}


def read_eval(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows[-1] if rows else {}


def count_data_rows(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)


def count_lines(path: Path):
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f if _.strip())


rows = []
for s in settings:
    timing = read_timing(s["out_base"], s["lm"])
    eval_path = s["setting"] / s["eval_name"]
    miss_path = s["setting"] / s["miss_name"]
    summary_path = s["setting"] / s["summary_name"]
    eval_row = read_eval(eval_path)
    instances = count_lines(s["setting"] / "instances.log")
    status = timing.get("status", "")
    if eval_row and status != "success":
        status = "post_eval_done"
    elif not status:
        status = "missing_or_incomplete"
    rows.append({
        "lm": s["lm"],
        "status": status,
        "eval_policy": s["eval_policy"],
        "samples": s["samples"],
        "sample_count": timing.get("sample_count", len(s["samples"].split())),
        "seconds": timing.get("seconds", ""),
        "minutes": timing.get("minutes", ""),
        "instances": instances,
        "BLEU": eval_row.get("BLEU", ""),
        "StreamLAAL": eval_row.get("StreamLAAL", ""),
        "StreamLAAL_CA": eval_row.get("StreamLAAL_CA", ""),
        "TERM_ACC": eval_row.get("TERM_ACC", ""),
        "TERM_CORRECT": eval_row.get("TERM_CORRECT", ""),
        "TERM_TOTAL": eval_row.get("TERM_TOTAL", ""),
        "miss_occurrences": count_data_rows(miss_path),
        "unique_missed_term_translations": count_data_rows(summary_path),
        "eval_tsv": str(eval_path),
        "misses_tsv": str(miss_path),
        "summary_tsv": str(summary_path),
        "output_dir": str(s["setting"]),
    })

out_tsv.parent.mkdir(parents=True, exist_ok=True)
with out_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Medicine zh Origin No-RAG Summary",
    "",
    f"- TSV: `{out_tsv}`",
    "- `lm=2` reuses the 20260522 generation but is re-scored with the current hard-manual glossary.",
    "- `lm=4` is the five-sample aggregate: Aries `404/545006/596001/606` plus Taurus orig80 `605000`.",
    "",
    "| lm | status | samples | BLEU | StreamLAAL | TERM_ACC | correct/total | misses | unique missed |",
    "|---:|---|---|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['lm']} | {r['status']} | {r['sample_count']} | "
        f"{r['BLEU'] or ''} | {r['StreamLAAL'] or ''} | {r['TERM_ACC'] or ''} | "
        f"{r['TERM_CORRECT']}/{r['TERM_TOTAL']} | {r['miss_occurrences']} | {r['unique_missed_term_translations']} |"
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"out_tsv": str(out_tsv), "out_md": str(out_md), "rows": rows}, ensure_ascii=False, indent=2))
PY

if command -v /home/jiaxuanluo/bin/codex-notify >/dev/null 2>&1; then
  /home/jiaxuanluo/bin/codex-notify --delay 8 --detach --workspace /home/jiaxuanluo/InfiniSST \
    "Codex collected: medicine zh origin no-RAG summary ${OUT_TSV}" || true
fi
