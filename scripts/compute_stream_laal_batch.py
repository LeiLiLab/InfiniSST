#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STREAM_LAAL_PATTERNS = [
    # Common variants observed in various forks/loggers
    re.compile(r"\bStreamLAAL\b\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    re.compile(r"\bstream_laal\b\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    # Fallback: sometimes printed as "LAAL (stream): 1.23"
    re.compile(r"\bLAAL\b.*?\bstream\b.*?[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
]


LANG_CONFIG = {
    "zh": {"tokenizer": "zh", "latency_unit": "char"},
    "ja": {"tokenizer": "ja-mecab", "latency_unit": "char"},
    "de": {"tokenizer": "13a", "latency_unit": "word"},
}


def parse_lang_instances(s: str) -> Tuple[str, str]:
    if ":" not in s:
        raise argparse.ArgumentTypeError("Expected 'lang:/path/to/instances.log'.")
    lang, path = s.split(":", 1)
    lang = lang.strip().lower()
    path = path.strip()
    if not lang or not path:
        raise argparse.ArgumentTypeError("Invalid 'lang:/path/to/instances.log'.")
    if lang not in LANG_CONFIG:
        raise argparse.ArgumentTypeError(f"Unsupported lang '{lang}'. Supported: {sorted(LANG_CONFIG.keys())}")
    return lang, path


def find_stream_laal(text: str) -> Optional[float]:
    for pat in STREAM_LAAL_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


@dataclass
class RunResult:
    lang: str
    instances: str
    glossary: str
    reference: str
    audio_yaml: str
    stream_laal: Optional[float]
    exit_code: int
    log_file: str


def run_one(
    stream_laal_py: str,
    lang: str,
    instances: str,
    glossary: str,
    reference: str,
    audio_yaml: str,
    out_log_dir: Path,
) -> RunResult:
    cfg = LANG_CONFIG[lang]
    cmd = [
        "python",
        stream_laal_py,
        "--simuleval-instances",
        instances,
        "--reference",
        reference,
        "--audio-yaml",
        audio_yaml,
        "--sacrebleu-tokenizer",
        cfg["tokenizer"],
        "--latency-unit",
        cfg["latency_unit"],
        "--glossary",
        glossary,
        "--term-lang",
        lang,
        "--term-mismatch-examples",
        "0",
    ]

    out_log_dir.mkdir(parents=True, exist_ok=True)
    safe_lang = re.sub(r"[^a-z0-9]+", "_", lang)
    safe_base = re.sub(r"[^a-z0-9]+", "_", Path(instances).parent.name.lower() or "run")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = out_log_dir / f"stream_laal_{safe_lang}_{safe_base}_{ts}.log"

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    output = proc.stdout or ""
    stream_laal = find_stream_laal(output)
    log_file.write_text(output, encoding="utf-8")

    return RunResult(
        lang=lang,
        instances=instances,
        glossary=glossary,
        reference=reference,
        audio_yaml=audio_yaml,
        stream_laal=stream_laal,
        exit_code=int(proc.returncode),
        log_file=str(log_file),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Batch compute StreamLAAL scores from SimulEval instances.log using stream_laal_term.py"
    )
    p.add_argument(
        "--instances",
        action="append",
        default=[],
        type=parse_lang_instances,
        help="Repeatable. Format: lang:/abs/path/to/instances.log (lang in {zh,ja,de}).",
    )
    p.add_argument(
        "--glossary",
        action="append",
        default=[],
        help="Repeatable. Glossary JSON path. If omitted, no runs will be executed.",
    )
    p.add_argument(
        "--audio-yaml",
        default="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml",
        help="Audio YAML path.",
    )
    p.add_argument(
        "--reference-template",
        default="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.{lang}.txt",
        help="Reference text template. Must include '{lang}'.",
    )
    p.add_argument(
        "--stream-laal-py",
        default=os.environ.get(
            "STREAM_LAAL_TERM_PY",
            "/home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py",
        ),
        help="Absolute path to stream_laal_term.py (or set env STREAM_LAAL_TERM_PY).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output CSV path.",
    )
    p.add_argument(
        "--log-dir",
        default="stream_laal_logs",
        help="Directory to write raw logs for each run.",
    )

    args = p.parse_args()

    if not args.instances:
        raise SystemExit("No --instances provided.")
    if not args.glossary:
        raise SystemExit("No --glossary provided. Add at least one --glossary JSON path.")
    if "{lang}" not in args.reference_template:
        raise SystemExit("--reference-template must contain '{lang}'.")

    stream_laal_py = str(args.stream_laal_py)
    if not os.path.isfile(stream_laal_py):
        raise SystemExit(f"stream_laal_term.py not found: {stream_laal_py}")

    out_log_dir = Path(args.log_dir).resolve()
    out_csv = Path(args.out).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    results: List[RunResult] = []
    for (lang, instances) in args.instances:
        ref = args.reference_template.format(lang=lang)
        for glossary in args.glossary:
            results.append(
                run_one(
                    stream_laal_py=stream_laal_py,
                    lang=lang,
                    instances=instances,
                    glossary=glossary,
                    reference=ref,
                    audio_yaml=args.audio_yaml,
                    out_log_dir=out_log_dir,
                )
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "lang",
                "instances",
                "glossary",
                "reference",
                "audio_yaml",
                "stream_laal",
                "exit_code",
                "log_file",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.lang,
                    r.instances,
                    r.glossary,
                    r.reference,
                    r.audio_yaml,
                    "" if r.stream_laal is None else f"{r.stream_laal:.6f}",
                    r.exit_code,
                    r.log_file,
                ]
            )

    # Minimal stdout summary (English)
    ok = sum(1 for r in results if r.exit_code == 0 and r.stream_laal is not None)
    print(f"[INFO] Wrote {len(results)} rows to {out_csv}")
    print(f"[INFO] Parsed StreamLAAL successfully for {ok}/{len(results)} runs")
    print(f"[INFO] Raw logs directory: {out_log_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









