#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


STREAM_LAAL_SCRIPT = "/home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"
DEFAULT_PYTHON_BIN = "/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python"
DEFAULT_MWER_ROOT = "/home/jiaxuanluo/mwerSegmenter"


@dataclass
class RunMetrics:
    lang: str
    run_dir: Path
    model: str
    chunk_size: float
    hop_size: float
    top_k: int
    voting_k: int
    bleu: Optional[float] = None
    stream_laal: Optional[float] = None
    stream_laal_ca: Optional[float] = None
    term_acc: Optional[float] = None
    term_correct: Optional[int] = None
    term_total: Optional[int] = None
    rtf_total: Optional[float] = None


def _infer_from_dirname(name: str) -> Tuple[str, float, float, int, int]:
    """
    Parse e.g.:
      best_model_no_llm_zh_v2_curated_cs1.92_hs0.48_H1_rk10_vk5
    """
    model = name.split("_curated_", 1)[0].split("_raw_", 1)[0]

    def m1(pat: str, cast):
        m = re.search(pat, name)
        if not m:
            raise ValueError(f"Cannot parse '{pat}' from {name}")
        return cast(m.group(1))

    cs = m1(r"_cs([0-9.]+)_", float)
    hs = m1(r"_hs([0-9.]+)_", float)
    rk = m1(r"_rk([0-9]+)_", int)
    vk = m1(r"_vk([0-9]+)$", int)
    return model, cs, hs, rk, vk


def _lang_settings(lang: str) -> Tuple[str, str]:
    if lang == "zh":
        return "zh", "char"
    if lang == "de":
        return "13a", "word"
    if lang == "ja":
        return "ja-mecab", "char"
    raise ValueError(f"Unsupported lang: {lang}")


def _parse_stream_laal_output(text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[int], Optional[int]]:
    # Find first line with 3 floats: BLEU StreamLAAL StreamLAAL_CA
    bleu = laal = laal_ca = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s+([0-9]+\.[0-9]+)\s*$", line)
        if m:
            bleu = float(m.group(1))
            laal = float(m.group(2))
            laal_ca = float(m.group(3))
            break

    term_acc = term_correct = term_total = None
    for line in text.splitlines():
        if line.startswith("TERM_ACC"):
            # Possible formats:
            # 1) TERM_ACC 0.8776 (86 / 98)
            # 2) TERM_ACC\t0.8776\tCORRECT_TERMS\t86\tTOTAL_TERMS\t98
            m = re.search(r"TERM_ACC\s+([0-9.]+)\s*.*\(\s*([0-9]+)\s*/\s*([0-9]+)\s*\)", line)
            if m:
                term_acc = float(m.group(1))
                term_correct = int(m.group(2))
                term_total = int(m.group(3))
                break
            parts = re.split(r"\s+", line.strip())
            # parts[0] == TERM_ACC
            if len(parts) >= 2:
                try:
                    term_acc = float(parts[1])
                except Exception:
                    term_acc = None
            if "CORRECT_TERMS" in parts:
                try:
                    term_correct = int(parts[parts.index("CORRECT_TERMS") + 1])
                except Exception:
                    term_correct = None
            if "TOTAL_TERMS" in parts:
                try:
                    term_total = int(parts[parts.index("TOTAL_TERMS") + 1])
                except Exception:
                    term_total = None
            break

    return bleu, laal, laal_ca, term_acc, term_correct, term_total


def _parse_rtf_from_simuleval_log(p: Path) -> Optional[float]:
    if not p.exists():
        return None
    txt = p.read_text(errors="ignore")
    ms = list(re.finditer(r"rtf_total=([0-9.]+)", txt))
    if not ms:
        return None
    return float(ms[-1].group(1))


def run_eval(
    run_dir: Path,
    lang: str,
    dataset_root: Path,
    glossary_path: Path,
    term_lang: str,
    python_bin: str,
    mwer_root: str,
) -> RunMetrics:
    name = run_dir.name
    model, cs, hs, rk, vk = _infer_from_dirname(name)
    tok, unit = _lang_settings(lang)

    instances = run_dir / "instances.log"
    if not instances.exists():
        raise FileNotFoundError(f"Missing instances.log: {instances}")

    ref_file = dataset_root / "dev/text/txt" / f"ACL.6060.dev.en-xx.{lang}.txt"
    audio_yaml = dataset_root / "dev.yaml"
    if not ref_file.exists():
        raise FileNotFoundError(f"Missing reference: {ref_file}")
    if not audio_yaml.exists():
        raise FileNotFoundError(f"Missing audio yaml: {audio_yaml}")
    if not glossary_path.exists():
        raise FileNotFoundError(f"Missing glossary: {glossary_path}")

    cmd = [
        python_bin,
        STREAM_LAAL_SCRIPT,
        "--simuleval-instances",
        str(instances),
        "--reference",
        str(ref_file),
        "--audio-yaml",
        str(audio_yaml),
        "--sacrebleu-tokenizer",
        tok,
        "--latency-unit",
        unit,
        "--glossary",
        str(glossary_path),
        "--term-lang",
        term_lang,
        "--term-mismatch-examples",
        "0",
    ]
    env = dict(**os.environ)
    # Required by stream_laal_term.py for mwerSegmenter.
    if mwer_root:
        env["MWERSEGMENTER_ROOT"] = mwer_root
        # Best-effort: make mwerSegmenter executable discoverable.
        env["PATH"] = f"{mwer_root}:{env.get('PATH','')}"
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    bleu, laal, laal_ca, term_acc, term_correct, term_total = _parse_stream_laal_output(out)

    m = RunMetrics(
        lang=lang,
        run_dir=run_dir,
        model=model,
        chunk_size=cs,
        hop_size=hs,
        top_k=rk,
        voting_k=vk,
        bleu=bleu,
        stream_laal=laal,
        stream_laal_ca=laal_ca,
        term_acc=term_acc,
        term_correct=term_correct,
        term_total=term_total,
        rtf_total=_parse_rtf_from_simuleval_log(run_dir / "simuleval.log"),
    )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, required=True, help="Base output dir, containing subfolders like zh/, de/")
    ap.add_argument("--dataset-root", type=str, default="/mnt/taurus/data/siqiouyang/datasets/acl6060")
    ap.add_argument("--glossary-path", type=str, default="/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json")
    ap.add_argument("--langs", type=str, default="zh,de", help="Comma-separated langs to evaluate, e.g. zh,de")
    ap.add_argument("--out-tsv", type=str, default="", help="Optional output TSV path. Default: <base-dir>/streamlaal_summary.tsv")
    ap.add_argument("--python-bin", type=str, default=DEFAULT_PYTHON_BIN, help="Python executable to run stream_laal_term.py")
    ap.add_argument("--mwer-root", type=str, default=DEFAULT_MWER_ROOT, help="MWERSEGMENTER_ROOT for stream_laal_term.py")
    args = ap.parse_args()

    base = Path(args.base_dir)
    dataset_root = Path(args.dataset_root)
    glossary_path = Path(args.glossary_path)
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    out_tsv = Path(args.out_tsv) if args.out_tsv else (base / "streamlaal_summary.tsv")

    rows: List[RunMetrics] = []
    for lang in langs:
        lang_dir = base / lang
        if not lang_dir.exists():
            continue
        for d in sorted([p for p in lang_dir.iterdir() if p.is_dir()]):
            # only consider dirs with instances.log
            if not (d / "instances.log").exists():
                continue
            rows.append(
                run_eval(
                    d,
                    lang=lang,
                    dataset_root=dataset_root,
                    glossary_path=glossary_path,
                    term_lang=lang,
                    python_bin=args.python_bin,
                    mwer_root=args.mwer_root,
                )
            )

    # write TSV
    header = [
        "lang",
        "model",
        "chunk_size",
        "hop_size",
        "top_k",
        "voting_k",
        "BLEU",
        "StreamLAAL",
        "StreamLAAL_CA",
        "TERM_ACC",
        "TERM_CORRECT",
        "TERM_TOTAL",
        "RTF",
        "run_dir",
    ]
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write(
                "\t".join(
                    [
                        r.lang,
                        r.model,
                        f"{r.chunk_size:.2f}",
                        f"{r.hop_size:.2f}",
                        str(r.top_k),
                        str(r.voting_k),
                        "" if r.bleu is None else f"{r.bleu:.3f}",
                        "" if r.stream_laal is None else f"{r.stream_laal:.3f}",
                        "" if r.stream_laal_ca is None else f"{r.stream_laal_ca:.3f}",
                        "" if r.term_acc is None else f"{r.term_acc:.4f}",
                        "" if r.term_correct is None else str(r.term_correct),
                        "" if r.term_total is None else str(r.term_total),
                        "" if r.rtf_total is None else f"{r.rtf_total:.3f}",
                        str(r.run_dir),
                    ]
                )
                + "\n"
            )

    print(f"[OK] Wrote: {out_tsv} (rows={len(rows)})")


if __name__ == "__main__":
    main()


