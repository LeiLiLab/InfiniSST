#!/usr/bin/env python3
"""
W&B Sweep agent for noisy_ratio exploration on 3-variant 1M data.

Sweeps over noisy_ratio while keeping other hyperparams fixed at
the best known values (lr=1e-4, temp=0.03, wiki_rank=1M).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import wandb

# ======Configuration=====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
CONDA_PREFIX = "/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
SAVE_DIR = "/mnt/data/jiaxuanluo/autoresearch_noisy_ratio"

TRAIN_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_train_3variant_1m.jsonl"
DEV_JSONL = "/mnt/gemini/data1/jiaxuanluo/term_dev_with_wiki_synth_normalized.jsonl"
ACL_DEV_JSONL = "/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval_extracted_paper_glossary/acl6060_dev_dataset.jsonl"
EVAL_WIKI_GLOSSARY = "/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/glossary_scale/wiki_glossary_nlp_ai_cs.json"

FIXED_ARGS = dict(
    epochs=1,
    batch_size=1024,
    num_workers=8,
    target_dim=1024,
    lora_rank=32,
    lora_alpha=64,
    text_lora_rank=128,
    text_lora_alpha=256,
    text_lr=0,
    lora_target_modules="q_proj k_proj v_proj out_proj fc1 fc2 proj1 proj2",
    text_lora_target_modules="query key value dense",
    glossary_neg_path="",
    glossary_neg_refresh_steps=0,
    neg_bank_size=0,
    neg_bank_refresh_steps=0,
    hard_neg_k=0,
    hard_neg_glossary="",
    save_steps=99999,
    eval_steps_sample=50,
    keep_checkpoints=1,
    eval_topk=10,
    eval_glossary_sizes="1000 10000",
    best_metric="eval_acl6060/recall@10_gs1000",
    best_metric_secondary="eval_acl6060/recall@10_gs10000",
    warmup_steps=200,
    time_budget=1500,
    lr=1e-4,
    temperature=0.03,
    wiki_rank=1000000,
)

NUM_GPUS = int(os.environ.get("SWEEP_NUM_GPUS", "2"))
# ======Configuration=====


def build_cmd(config: dict) -> list[str]:
    noisy_ratio = config.get("noisy_ratio", 1.0)
    tag = f"nr{noisy_ratio:.2f}"
    save_path = os.path.join(SAVE_DIR, f"{tag}.pt")

    master_port = 29922 + int(os.environ.get("SWEEP_AGENT_IDX", "0"))

    cmd = [
        f"{CONDA_PREFIX}/bin/torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        TRAIN_PY,
        "--train_jsonl", TRAIN_JSONL,
        "--dev_jsonl", DEV_JSONL,
        "--save_path", save_path,
        "--noisy_ratio", str(noisy_ratio),
        "--acl_dev_jsonl", ACL_DEV_JSONL,
        "--eval_wiki_glossary", EVAL_WIKI_GLOSSARY,
        "--use_lora",
        "--enable_wandb",
        "--wandb_project", "qwen3_rag_autoresearch",
        "--wandb_exp_name", tag,
    ]

    for k, v in FIXED_ARGS.items():
        if k in ("lora_target_modules", "text_lora_target_modules", "eval_glossary_sizes"):
            cmd.append(f"--{k}")
            cmd.extend(str(v).split())
        else:
            cmd.extend([f"--{k}", str(v)])

    return cmd


def stream_and_report(proc: subprocess.Popen, run: wandb.sdk.wandb_run.Run) -> str:
    all_stderr = []
    step_pattern = re.compile(r"\[EVAL_ACL6060\] step=(\d+).*gs10000.*r@10=([\d.]+)")
    gs1k_pattern = re.compile(r"\[EVAL_ACL6060\] step=(\d+).*gs1000.*r@10=([\d.]+)")
    best_gs10k = 0.0
    best_gs1k = 0.0

    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
        all_stderr.append(line)

        m = step_pattern.search(line)
        if m:
            step = int(m.group(1))
            gs10k = float(m.group(2))
            if gs10k > best_gs10k:
                best_gs10k = gs10k
            run.log({"best_acl6060_r10_gs10k": best_gs10k, "step": step}, step=step)

        m2 = gs1k_pattern.search(line)
        if m2:
            step = int(m2.group(1))
            gs1k = float(m2.group(2))
            if gs1k > best_gs1k:
                best_gs1k = gs1k
            run.log({"best_acl6060_r10_gs1k": best_gs1k, "step": step}, step=step)

    return "".join(all_stderr)


def parse_best_metrics(stderr_text: str) -> dict:
    best = {"best_acl6060_r10_gs1k": 0.0, "best_acl6060_r10_gs10k": 0.0}
    for line in stderr_text.split("\n"):
        m = re.search(r"\[BEST\].*recall@10_gs1000=([\d.]+)", line)
        if m:
            best["best_acl6060_r10_gs1k"] = max(best["best_acl6060_r10_gs1k"], float(m.group(1)))
        m2 = re.search(r"\[BEST_SECONDARY\].*recall@10_gs10000=([\d.]+)", line)
        if m2:
            best["best_acl6060_r10_gs10k"] = max(best["best_acl6060_r10_gs10k"], float(m2.group(1)))
    return best


def main():
    run = wandb.init()
    assert run is not None
    config = dict(wandb.config)
    print(f"[SWEEP] Config: {json.dumps(config, indent=2)}", flush=True)

    os.makedirs(SAVE_DIR, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"/mnt/taurus/home/jiaxuanluo/InfiniSST:{env.get('PYTHONPATH', '')}"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["NCCL_TIMEOUT"] = "7200"
    env["NCCL_P2P_DISABLE"] = "1"
    env["WANDB_MODE"] = "disabled"

    cmd = build_cmd(config)
    print(f"[SWEEP] Running: {' '.join(cmd[:5])} ...", flush=True)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, env=env,
    )

    stderr_text = stream_and_report(proc, run)
    proc.wait()

    best = parse_best_metrics(stderr_text)
    run.summary["best_acl6060_r10_gs1k"] = best["best_acl6060_r10_gs1k"]
    run.summary["best_acl6060_r10_gs10k"] = best["best_acl6060_r10_gs10k"]
    run.summary["exit_code"] = proc.returncode

    print(
        f"[SWEEP] Done. exit={proc.returncode} "
        f"best_gs1k={best['best_acl6060_r10_gs1k']:.4f} "
        f"best_gs10k={best['best_acl6060_r10_gs10k']:.4f}",
        flush=True,
    )
    run.finish()


if __name__ == "__main__":
    main()
