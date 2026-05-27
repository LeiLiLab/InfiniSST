#!/usr/bin/env python3
"""Compare a batched-vLLM RAG eval run against a serial SimulEval run.

The goal is not to prove exact token identity.  vLLM sampling can still diverge
when execution order changes.  This tool checks whether the inputs, retrieval
records, output lengths, delay traces, and final metrics are close enough before
batch outputs are treated as a drop-in accelerated path.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence


TERM_RE = re.compile(r"</?(?:term|t)>", re.IGNORECASE)
BAD_PREFIX_RE = re.compile(r"^<term>(?:term>){3,}", re.IGNORECASE)


def _load_eval(path: Path) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"Expected one eval row in {path}, got {len(rows)}")
    return rows[0]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid json: {exc}") from exc
    return rows


def _records(rows: Sequence[Dict[str, Any]], typ: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("type") == typ]


def _ref_terms(rec: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for ref in rec.get("references") or []:
        term = str(ref.get("term") or "").strip()
        if term:
            out.append(term)
    return out


def _prompt_sig(rec: Dict[str, Any]) -> str:
    text = str(rec.get("prompt") or "")
    return re.sub(r"\s+", " ", text).strip()


def _runtime_summary(path: Path) -> Dict[str, Any]:
    rows = _load_jsonl(path)
    inputs = _records(rows, "llm_input")
    outputs = _records(rows, "llm_output")
    ref_hist = Counter(len(_ref_terms(row)) for row in inputs)
    return {
        "path": str(path),
        "rows": len(rows),
        "llm_inputs": len(inputs),
        "llm_outputs": len(outputs),
        "avg_refs_per_input": mean([len(_ref_terms(row)) for row in inputs]) if inputs else 0.0,
        "nonempty_ref_rate": (
            sum(1 for row in inputs if _ref_terms(row)) / len(inputs) if inputs else 0.0
        ),
        "ref_hist": dict(sorted(ref_hist.items())),
        "inputs": inputs,
        "outputs": outputs,
    }


def _instances_summary(path: Path) -> Dict[str, Any]:
    rows = _load_jsonl(path)
    pred_chars = [len(str(row.get("prediction") or "")) for row in rows]
    delay_lengths = [len(row.get("delays") or []) for row in rows]
    tag_spans = [len(TERM_RE.findall(str(row.get("prediction") or ""))) // 2 for row in rows]
    bad_prefix = sum(1 for row in rows if BAD_PREFIX_RE.search(str(row.get("prediction") or "")))
    return {
        "path": str(path),
        "instances": len(rows),
        "prediction_chars": sum(pred_chars),
        "avg_prediction_chars": mean(pred_chars) if pred_chars else 0.0,
        "delay_units": sum(delay_lengths),
        "avg_delay_units": mean(delay_lengths) if delay_lengths else 0.0,
        "term_tag_spans": sum(tag_spans),
        "bad_term_prefix_instances": bad_prefix,
        "rows": rows,
    }


def _metric_float(row: Dict[str, str], key: str) -> Optional[float]:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _first_mismatch(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]], key: str) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if str(a[i].get(key) or "") != str(b[i].get(key) or ""):
            return i
    if len(a) != len(b):
        return n
    return None


def _prompt_mismatch_note(batch_row: Dict[str, Any], serial_row: Dict[str, Any]) -> str:
    batch_prompt = str(batch_row.get("prompt") or "")
    serial_prompt = str(serial_row.get("prompt") or "")
    marker = "\n...[truncated]..."
    if marker in batch_prompt:
        prefix = batch_prompt.split(marker, 1)[0]
        if serial_prompt.startswith(prefix):
            return (
                "batch runtime prompt is truncated for logging only; "
                f"logged_prefix_chars={len(prefix)}, "
                f"batch_full_prompt_chars={batch_row.get('prompt_chars')}, "
                f"serial_prompt_chars={len(serial_prompt)}"
            )
    return "raw prompt strings differ"


def _format_metrics(batch: Dict[str, str], serial: Dict[str, str], keys: Sequence[str]) -> List[str]:
    lines = [
        "| metric | serial | batch | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in keys:
        s = _metric_float(serial, key)
        b = _metric_float(batch, key)
        if s is None or b is None:
            continue
        lines.append(f"| {key} | {s:.6g} | {b:.6g} | {b - s:+.6g} |")
    return lines


def _format_dict(d: Dict[Any, Any]) -> str:
    return ", ".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: kv[0]))


def build_report(args: argparse.Namespace) -> str:
    batch_eval = _load_eval(Path(args.batch_eval))
    serial_eval = _load_eval(Path(args.serial_eval))
    batch_rt = _runtime_summary(Path(args.batch_runtime))
    serial_rt = _runtime_summary(Path(args.serial_runtime))
    batch_inst = _instances_summary(Path(args.batch_instances))
    serial_inst = _instances_summary(Path(args.serial_instances))

    batch_inputs = batch_rt.pop("inputs")
    serial_inputs = serial_rt.pop("inputs")
    batch_outputs = batch_rt.pop("outputs")
    serial_outputs = serial_rt.pop("outputs")
    batch_instance_rows = batch_inst.pop("rows")
    serial_instance_rows = serial_inst.pop("rows")

    prompt_mismatch = _first_mismatch(batch_inputs, serial_inputs, "prompt")
    output_mismatch = _first_mismatch(batch_outputs, serial_outputs, "text")
    ref_count_mismatch = None
    for i, (b_row, s_row) in enumerate(zip(batch_inputs, serial_inputs)):
        if len(_ref_terms(b_row)) != len(_ref_terms(s_row)):
            ref_count_mismatch = i
            break
    if ref_count_mismatch is None and len(batch_inputs) != len(serial_inputs):
        ref_count_mismatch = min(len(batch_inputs), len(serial_inputs))

    instance_prediction_mismatch = _first_mismatch(
        batch_instance_rows, serial_instance_rows, "prediction"
    )
    instance_delay_mismatch = None
    instance_elapsed_mismatch = None
    for i, (b_row, s_row) in enumerate(zip(batch_instance_rows, serial_instance_rows)):
        if b_row.get("delays") != s_row.get("delays") and instance_delay_mismatch is None:
            instance_delay_mismatch = i
        if b_row.get("elapsed") != s_row.get("elapsed") and instance_elapsed_mismatch is None:
            instance_elapsed_mismatch = i
        if instance_delay_mismatch is not None and instance_elapsed_mismatch is not None:
            break
    if instance_delay_mismatch is None and len(batch_instance_rows) != len(serial_instance_rows):
        instance_delay_mismatch = min(len(batch_instance_rows), len(serial_instance_rows))
    if instance_elapsed_mismatch is None and len(batch_instance_rows) != len(serial_instance_rows):
        instance_elapsed_mismatch = min(len(batch_instance_rows), len(serial_instance_rows))

    lines: List[str] = []
    lines.append("# Batched vLLM vs Serial Alignment Report")
    lines.append("")
    lines.append("## Metrics")
    lines.extend(
        _format_metrics(
            batch_eval,
            serial_eval,
            ["BLEU", "TERM_ACC", "REAL_TERM_ADOPT", "TERM_FCR", "StreamLAAL", "StreamLAAL_CA"],
        )
    )
    lines.append("")
    lines.append("## Runtime Input Alignment")
    lines.append("")
    lines.append("| field | serial | batch |")
    lines.append("| --- | ---: | ---: |")
    for key in ["rows", "llm_inputs", "llm_outputs", "avg_refs_per_input", "nonempty_ref_rate"]:
        lines.append(f"| {key} | {serial_rt[key]} | {batch_rt[key]} |")
    lines.append(f"| ref_hist | `{_format_dict(serial_rt['ref_hist'])}` | `{_format_dict(batch_rt['ref_hist'])}` |")
    lines.append("")
    lines.append(f"- first prompt mismatch index: `{prompt_mismatch}`")
    lines.append(f"- first output mismatch index: `{output_mismatch}`")
    lines.append(f"- first reference-count mismatch index: `{ref_count_mismatch}`")
    if prompt_mismatch is not None and prompt_mismatch < len(batch_inputs) and prompt_mismatch < len(serial_inputs):
        lines.append(
            f"- prompt mismatch note: `{_prompt_mismatch_note(batch_inputs[prompt_mismatch], serial_inputs[prompt_mismatch])}`"
        )
        lines.append("")
        lines.append("### Prompt Mismatch Snippet")
        lines.append("")
        lines.append("```text")
        lines.append("[serial]")
        lines.append(_prompt_sig(serial_inputs[prompt_mismatch])[: args.snippet_chars])
        lines.append("[batch]")
        lines.append(_prompt_sig(batch_inputs[prompt_mismatch])[: args.snippet_chars])
        lines.append("```")
    if output_mismatch is not None and output_mismatch < len(batch_outputs) and output_mismatch < len(serial_outputs):
        lines.append("")
        lines.append("### Output Mismatch Snippet")
        lines.append("")
        lines.append("```text")
        lines.append("[serial]")
        lines.append(str(serial_outputs[output_mismatch].get("text") or "")[: args.snippet_chars])
        lines.append("[batch]")
        lines.append(str(batch_outputs[output_mismatch].get("text") or "")[: args.snippet_chars])
        lines.append("```")
    lines.append("")
    lines.append("## Instance / Delay Summary")
    lines.append("")
    lines.append("| field | serial | batch |")
    lines.append("| --- | ---: | ---: |")
    for key in [
        "instances",
        "prediction_chars",
        "avg_prediction_chars",
        "delay_units",
        "avg_delay_units",
        "term_tag_spans",
        "bad_term_prefix_instances",
    ]:
        lines.append(f"| {key} | {serial_inst[key]} | {batch_inst[key]} |")
    lines.append("")
    lines.append(f"- first instance prediction mismatch index: `{instance_prediction_mismatch}`")
    lines.append(f"- first instance delay mismatch index: `{instance_delay_mismatch}`")
    lines.append(f"- first instance elapsed mismatch index: `{instance_elapsed_mismatch}`")
    if (
        instance_prediction_mismatch is not None
        and instance_prediction_mismatch < len(batch_instance_rows)
        and instance_prediction_mismatch < len(serial_instance_rows)
    ):
        idx = instance_prediction_mismatch
        lines.append("")
        lines.append("### Instance Prediction Mismatch Snippet")
        lines.append("")
        lines.append("```text")
        lines.append("[serial]")
        lines.append(str(serial_instance_rows[idx].get("prediction") or "")[: args.snippet_chars])
        lines.append("[batch]")
        lines.append(str(batch_instance_rows[idx].get("prediction") or "")[: args.snippet_chars])
        lines.append("```")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- serial eval: `{args.serial_eval}`")
    lines.append(f"- batch eval: `{args.batch_eval}`")
    lines.append(f"- serial runtime: `{args.serial_runtime}`")
    lines.append(f"- batch runtime: `{args.batch_runtime}`")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serial-eval", required=True)
    p.add_argument("--batch-eval", required=True)
    p.add_argument("--serial-runtime", required=True)
    p.add_argument("--batch-runtime", required=True)
    p.add_argument("--serial-instances", required=True)
    p.add_argument("--batch-instances", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--snippet-chars", type=int, default=1200)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out) + ".tmp")
    tmp.write_text(report + "\n", encoding="utf-8")
    tmp.replace(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
