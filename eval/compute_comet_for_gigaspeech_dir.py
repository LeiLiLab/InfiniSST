#!/usr/bin/env python3
# from __future__ imports must be at the beginning (after shebang/comments).
from __future__ import annotations

# ======Configuration=====
DEFAULT_MODEL_NAME = "Unbabel/wmt22-comet-da"
DEFAULT_BATCH_SIZE = 4
DEFAULT_TOP_N = 10
DEFAULT_MAX_HEADER_LINES_TO_SCAN = 50
DEFAULT_SRC_FULL = "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
DEFAULT_AUDIO_YAML_FULL = "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml"
# ======Configuration=====

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


def _first_non_empty_line(path: Path, max_lines: int) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            if line.strip():
                return line
    return ""


def _read_bleu_from_scores_tsv(scores_tsv: Path) -> float | None:
    if not scores_tsv.exists():
        return None
    try:
        with scores_tsv.open("r", encoding="utf-8") as f:
            _ = f.readline()
            row = f.readline()
        if not row:
            return None
        return float(row.split("\t")[0])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute COMET/XCOMET for InfiniSST run folders containing instances.log."
    )
    ap.add_argument("--run-root", type=str, required=True, help="Directory containing run subfolders.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="COMET model repo id or name.")
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="If set, load model from a local .ckpt path instead of downloading from HuggingFace.",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--max-header-lines", type=int, default=DEFAULT_MAX_HEADER_LINES_TO_SCAN)
    ap.add_argument("--gpus", type=int, default=1, help="Number of GPUs for COMET predict (0 for CPU).")
    ap.add_argument(
        "--cuda-visible-devices",
        type=str,
        default="0",
        help="Set CUDA_VISIBLE_DEVICES inside the process (only works if not readonly in your shell).",
    )
    ap.add_argument(
        "--src-full",
        type=str,
        default=DEFAULT_SRC_FULL,
        help="Full dev source transcript file (English). Used to construct src for each paper wav.",
    )
    ap.add_argument(
        "--audio-yaml-full",
        type=str,
        default=DEFAULT_AUDIO_YAML_FULL,
        help="Full dev.yaml containing wav field for each segment.",
    )
    args = ap.parse_args()

    run_root = Path(args.run_root)
    if not run_root.exists():
        raise SystemExit(f"Run root not found: {run_root}")

    # Ensure HF token is visible to huggingface_hub (if provided)
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]

    # If the parent shell enforces CUDA_VISIBLE_DEVICES as readonly, run this script via:
    #   env -u CUDA_VISIBLE_DEVICES python ...
    # so we can set it here.
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    from comet import download_model, load_from_checkpoint

    name_re = re.compile(r"2022\.acl-long\.(\d+)")
    cs_re = re.compile(r"_cs([0-9.]+)_")
    lm_re = re.compile(r"_lm(\d+)_")

    # Load full source transcript and dev.yaml once, to build per-paper src text.
    src_full_path = Path(args.src_full)
    audio_yaml_full_path = Path(args.audio_yaml_full)
    if not src_full_path.exists():
        raise SystemExit(f"Missing src-full file: {src_full_path}")
    if not audio_yaml_full_path.exists():
        raise SystemExit(f"Missing audio-yaml-full file: {audio_yaml_full_path}")

    try:
        import yaml  # type: ignore
    except Exception as e:
        raise SystemExit(f"Missing dependency: pyyaml (yaml). Error: {type(e).__name__}: {e}")

    src_full_lines = src_full_path.read_text(encoding="utf-8", errors="replace").splitlines()
    audio_yaml_full = yaml.safe_load(audio_yaml_full_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(audio_yaml_full, list):
        raise SystemExit("audio-yaml-full must be a YAML list.")
    if len(audio_yaml_full) != len(src_full_lines):
        raise SystemExit(
            f"Length mismatch: dev.yaml entries={len(audio_yaml_full)} != src-full lines={len(src_full_lines)}"
        )

    def _paper_src_text(paper_id: str) -> str:
        wav_name = f"{paper_id}.wav"
        idxs = [
            i
            for i, x in enumerate(audio_yaml_full)
            if isinstance(x, dict) and x.get("wav") == wav_name
        ]
        # Keep line breaks to preserve segmentation.
        return "\n".join(src_full_lines[i] for i in idxs).strip()

    run_dirs: List[Path] = []
    for p in run_root.iterdir():
        if not p.is_dir():
            continue
        if p.name == "__paper_inputs__":
            continue
        if (p / "instances.log").exists():
            run_dirs.append(p)
    run_dirs.sort(key=lambda x: x.name)

    items: List[Dict[str, str]] = []
    meta: List[Dict[str, Any]] = []
    skipped: List[Tuple[str, str]] = []

    for d in run_dirs:
        inst_path = d / "instances.log"
        line = _first_non_empty_line(inst_path, max_lines=args.max_header_lines)
        if not line:
            skipped.append((d.name, "empty"))
            continue

        try:
            obj = json.loads(line)
        except Exception as e:
            skipped.append((d.name, f"json_error:{type(e).__name__}"))
            continue

        pred = (obj.get("prediction") or "").strip()
        ref = (obj.get("reference") or "").strip()
        if not pred or not ref:
            skipped.append((d.name, "missing_pred_or_ref"))
            continue

        m_id = name_re.search(d.name)
        paper_id = m_id.group(0) if m_id else "unknown"
        cs = cs_re.search(d.name)
        lm = lm_re.search(d.name)

        src_text = ""
        if paper_id != "unknown":
            src_text = _paper_src_text(paper_id)

        meta.append(
            {
                "dir": d.name,
                "paper": paper_id,
                "cs": cs.group(1) if cs else "",
                "lm": lm.group(1) if lm else "",
                "bleu": _read_bleu_from_scores_tsv(d / "scores.tsv"),
            }
        )

        items.append({"src": src_text, "mt": pred, "ref": ref})

    print(f"[INFO] run_dirs={len(run_dirs)} usable_items={len(items)} skipped={len(skipped)}")
    if skipped:
        print("[INFO] skipped (up to 20):")
        for n, r in skipped[:20]:
            print(f"  - {n}: {r}")
    if not items:
        raise SystemExit("No usable items to score.")

    ckpt_path = args.checkpoint.strip()
    if ckpt_path:
        print(f"[INFO] Loading model from local checkpoint: {ckpt_path}")
        model = load_from_checkpoint(ckpt_path)
    else:
        ckpt_path = download_model(args.model)
        print(f"[INFO] Model checkpoint: {ckpt_path}")
        model = load_from_checkpoint(ckpt_path)

    if int(args.gpus) > 0:
        model.to("cuda")
    else:
        model.to("cpu")

    out = model.predict(items, batch_size=int(args.batch_size), gpus=int(args.gpus))
    scores = list(map(float, out.scores))

    model_tag = args.model if not args.checkpoint else ckpt_path
    print(f"\n[RESULT] {model_tag} system_score (mean over items): {out.system_score * 100:.2f}")
    print("[NOTE] src is built from ACL.6060.dev.en-xx.en.txt by concatenating segments for each paper wav.")

    bucket: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for m, s in zip(meta, scores):
        bucket[(m["cs"], m["lm"])].append(s)

    print("\n[BY_CONFIG] mean_score*100 (n)")
    for (cs_v, lm_v), ss in sorted(
        bucket.items(),
        key=lambda kv: (float(kv[0][0] or 0.0), int(kv[0][1] or 0)),
    ):
        print(f"  cs={cs_v:>4} lm={lm_v:>2}: {mean(ss) * 100:.2f} (n={len(ss)})")

    ranked = sorted(zip(meta, scores), key=lambda x: x[1], reverse=True)
    print("\n[TOP_RUNS] score*100 | bleu | cs lm | paper | dir")
    for m, s in ranked[: int(args.top_n)]:
        b = "NA" if m["bleu"] is None else f"{m['bleu']:.2f}"
        print(
            f"  {s*100:.2f} | {b:>5} | cs={m['cs']} lm={m['lm']} | {m['paper']} | {m['dir']}"
        )


if __name__ == "__main__":
    main()

