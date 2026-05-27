#!/usr/bin/env python3
"""Re-score PSC medicine gs outputs against the canonical local raw denominator.

The PSC generation jobs used the correct fixed raw glossary file, but their
staged reference text differs from the local medicine raw main-result reference.
This script fetches PSC instances/runtime logs and re-runs the offline scorer
with the same source/ref/glossary files used by the medicine raw main result.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path("/home/jiaxuanluo/InfiniSST")
LOCAL_RAW_INPUTS = Path(
    "/mnt/gemini/data1/jiaxuanluo/medicine_hardraw_hn1024_tau078_new_v9_batch_"
    "20260524T0242/zh/__medicine_inputs__/lists"
)
DEFAULT_OUT_ROOT = Path("/mnt/gemini/data1/jiaxuanluo/psc_medicine_gs_reposteval_fixedraw_20260525")

REMOTE_ROOTS = [
    "/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/"
    "medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/"
    "20260524T1837_retry5h_audio8trim_psc_med_newv9_hn1024_tau078_"
    "gs1k_gs10k_lm12_fixedraw_zh",
    "/ocean/projects/cis260009p/jluo7/InfiniSST_psc_eval/outputs/"
    "medicine_hardraw_new_v9_hn1024_tau078_gs_fixedraw/"
    "20260524T1345_retry_audio8trim_psc_med_newv9_hn1024_tau078_"
    "gs1k_gs10k_fixedraw_zh",
]

SSH_TARGET = "jluo7@bridges2.psc.edu"
SSH_CONTROL = "/home/jiaxuanluo/.ssh/sockets/jluo7@bridges2.psc.edu:22"
SSH_BASE_CMD = [
    "ssh",
    "-S",
    SSH_CONTROL,
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=2",
    SSH_TARGET,
]
BANKS = ("gs1k", "gs10k")
LMS = (1, 2, 3, 4)


def ssh_text(command: str) -> str:
    proc = subprocess.run(
        [*SSH_BASE_CMD, command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=150,
    )
    return proc.stdout


def remote_output_dir(bank: str, lm: int) -> str | None:
    for root in REMOTE_ROOTS:
        base = f"{root}/{bank}/lm{lm}/zh"
        cmd = (
            "for d in "
            f"{shlex.quote(base)}/*; do "
            '[ -s "$d/eval_results.tsv" ] && [ -s "$d/instances.log" ] '
            '&& { printf "%s\\n" "$d"; break; }; '
            "done; true"
        )
        out = ssh_text(cmd).strip()
        if out:
            return out
    return None


def remote_runtime_log(remote_dir: str) -> str:
    cmd = (
        "ls -1 "
        f"{shlex.quote(remote_dir)}/runtime_omni_vllm_maxsim_rag_*.jsonl "
        "2>/dev/null | tail -1"
    )
    out = ssh_text(cmd).strip()
    if not out:
        raise RuntimeError(f"missing runtime JSONL under {remote_dir}")
    return out


def fetch_remote(remote_path: str, local_path: Path, *, force: bool) -> None:
    if local_path.is_file() and local_path.stat().st_size > 0 and not force:
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("wb") as f:
        subprocess.run(
            [*SSH_BASE_CMD, f"cat {shlex.quote(remote_path)}"],
            check=True,
            stdout=f,
            timeout=300,
        )


def run_posteval(local_dir: Path, out_tsv: Path) -> None:
    source_text = LOCAL_RAW_INPUTS / "medicine.source_text.en__medicine5_hardraw.txt"
    ref_file = LOCAL_RAW_INPUTS / "medicine.ref.zh__medicine5_hardraw.txt"
    audio_yaml = LOCAL_RAW_INPUTS / "medicine.audio__medicine5_hardraw.yaml"
    glossary = LOCAL_RAW_INPUTS / "hard_medicine_raw__medicine5.json"
    for path in [source_text, ref_file, audio_yaml, glossary, local_dir / "instances.log"]:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    env = dict(**__import__("os").environ)
    env["MWERSEGMENTER_ROOT"] = "/home/jiaxuanluo/mwerSegmenter"
    env["PATH"] = f"/home/jiaxuanluo/mwerSegmenter:{env.get('PATH', '')}"
    subprocess.run(
        [
            "python",
            str(ROOT / "documents/code/offline_sst_eval/offline_streamlaal_eval.py"),
            "--mode",
            "acl6060",
            "--instances-log",
            str(local_dir / "instances.log"),
            "--lang-code",
            "zh",
            "--source-file",
            str(source_text),
            "--ref-file",
            str(ref_file),
            "--audio-yaml",
            str(audio_yaml),
            "--glossary-acl6060",
            str(glossary),
            "--fbk-fairseq-root",
            "/mnt/taurus/home/jiaxuanluo/FBK-fairseq",
            "--python-bin",
            "/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python",
            "--strip-output-tags",
            "term",
            "--term-fcr-policy",
            "term_map_source_ref_negative_sentence",
            "--output-tsv",
            str(out_tsv),
            "--output-log",
            str(local_dir / "eval_results.localraw.log"),
        ],
        cwd=str(ROOT),
        env=env,
        check=True,
    )


def iter_targets(only: Iterable[str] | None) -> Iterable[tuple[str, int]]:
    selected = set(only or [])
    for bank in BANKS:
        for lm in LMS:
            key = f"{bank}_lm{lm}"
            if selected and key not in selected:
                continue
            yield bank, lm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--force-fetch", action="store_true")
    ap.add_argument("--force-posteval", action="store_true")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--only", nargs="*", help="Optional keys like gs1k_lm1 gs10k_lm2")
    args = ap.parse_args()

    missing: list[str] = []
    completed: list[str] = []
    for bank, lm in iter_targets(args.only):
        key = f"{bank}_lm{lm}"
        local_dir = args.out_root / key
        out_tsv = local_dir / "eval_results.localraw.tsv"
        remote_dir = remote_output_dir(bank, lm)
        if not remote_dir:
            missing.append(key)
            print(f"[MISS] {key}: no remote eval_results.tsv yet")
            continue
        runtime_log = remote_runtime_log(remote_dir)
        fetch_remote(f"{remote_dir}/instances.log", local_dir / "instances.log", force=args.force_fetch)
        fetch_remote(runtime_log, local_dir / Path(runtime_log).name, force=args.force_fetch)
        (local_dir / "remote_output_dir.txt").write_text(remote_dir + "\n", encoding="utf-8")
        if not out_tsv.is_file() or args.force_posteval:
            run_posteval(local_dir, out_tsv)
        completed.append(key)
        print(f"[OK] {key}: {out_tsv}")

    if missing and not args.allow_missing:
        raise SystemExit("missing remote rows: " + ", ".join(missing))
    print(f"[DONE] reposteval complete={len(completed)} missing={len(missing)} out_root={args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
