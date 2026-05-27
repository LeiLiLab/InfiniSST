#!/usr/bin/env python3
"""Build canonical main-result TSVs and paper figures.

This script intentionally keeps two sources separate:

* user-supplied reusable rows for historical offline / InfiniSST / paper-glossary
  results;
* verified filesystem artifacts for the new tagged-raw and medicine RASST rows.

The generated TSV is the source for the two paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path("/home/jiaxuanluo/InfiniSST")
REPORT_DIR = ROOT / "documents/code/simuleval/reports"
PAPER_FIG_DIR = (
    ROOT
    / "documents/code/train/term_train/reports/EMNLP_26_InfiniSST_RAG_src/latex/figures"
)
USER_SOURCE = "user_prompt_2026-05-24"
OFFLINE_LLM_ROOT = Path("/mnt/data/siqiouyang/runs/infinisst_rag/offline")
ACL_SOURCE_FILE = Path(
    "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.en.txt"
)
ACL_TAGGED_RAW_GLOSSARY = Path(
    "/mnt/gemini/home/jiaxuanluo/eval_glossaries/acl6060_tagged_gt_raw_min_norm2.json"
)
MEDICINE_SOURCE_FILE = Path(
    "/mnt/gemini/data1/jiaxuanluo/"
    "medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm1_aries01/"
    "zh/__medicine_inputs__/combined/medicine5.source_text.en.sentences.txt"
)
MEDICINE_HARDRAW_GLOSSARY = Path(
    "/mnt/gemini/home/jiaxuanluo/medicine_eval_hard_terms_llm_judge_manual_20260524/"
    "hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json"
)

LANGS: Sequence[Tuple[str, str]] = (("zh", "En-Zh"), ("de", "En-De"), ("ja", "En-Ja"))
LMS: Sequence[int] = (1, 2, 3, 4)
FIELDS = [
    "dataset",
    "method",
    "lang",
    "lm",
    "BLEU",
    "StreamLAAL",
    "StreamLAAL_CA",
    "TERM_ACC",
    "TERM_CORRECT",
    "TERM_TOTAL",
    "source_type",
    "source_path",
    "event_id",
    "wandb_run_id",
    "status",
    "note",
]


@dataclass(frozen=True)
class EvalSource:
    path: Path
    event_id: str
    wandb_run_id: str = ""
    source_type: str = "verified_eval_results"
    status: str = "verified"
    note: str = ""


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def row(
    *,
    dataset: str,
    method: str,
    lang: str,
    lm: int | str | None,
    bleu: object = None,
    streamlaal: object = None,
    streamlaal_ca: object = None,
    term_acc: object = None,
    term_correct: object = None,
    term_total: object = None,
    source_type: str,
    source_path: str,
    event_id: str = "",
    wandb_run_id: str = "",
    status: str,
    note: str = "",
) -> Dict[str, str]:
    return {
        "dataset": dataset,
        "method": method,
        "lang": lang,
        "lm": "NA" if lm is None else str(lm),
        "BLEU": fmt(bleu),
        "StreamLAAL": fmt(streamlaal),
        "StreamLAAL_CA": fmt(streamlaal_ca),
        "TERM_ACC": fmt(term_acc),
        "TERM_CORRECT": fmt(term_correct, digits=0),
        "TERM_TOTAL": fmt(term_total, digits=0),
        "source_type": source_type,
        "source_path": source_path,
        "event_id": event_id,
        "wandb_run_id": wandb_run_id,
        "status": status,
        "note": note,
    }


def read_last_tsv_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty TSV: {path}")
    return dict(rows[-1])


def read_single_tsv_row(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"expected one data row in {path}, found {len(rows)}")
    return dict(rows[0])


def read_text_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def normalise_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def source_contains(source_text: str, term: str) -> bool:
    source_norm = normalise_space(source_text).casefold()
    term_norm = normalise_space(term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def load_streamlaal_terms(glossary_path: Path, lang: str) -> List[Dict[str, str]]:
    """Load terms with the same target-translation dedup rule as stream_laal_term.py."""
    data = json.loads(glossary_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        iterable: Iterable[Tuple[object, object]] = data.items()
    elif isinstance(data, list):
        iterable = enumerate(data)
    else:
        raise ValueError(f"unsupported glossary format: {glossary_path}")

    terms: Dict[str, str] = {}
    for key, entry in iterable:
        if not isinstance(entry, dict):
            continue
        translations = entry.get("target_translations") or {}
        if not isinstance(translations, dict):
            continue
        target = str(translations.get(lang) or "").strip()
        if not target:
            continue
        source_term = str(entry.get("term") or key)
        if target not in terms:
            terms[target] = source_term
    return [{"target": target, "en": source} for target, source in sorted(terms.items())]


def read_offline_instances(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError(f"non-object JSONL row in {path}")
                rows.append(obj)
    if not rows:
        raise ValueError(f"empty offline instances log: {path}")
    return rows


def score_offline_term_accuracy(
    *,
    instances_log: Path,
    source_file: Path,
    glossary_path: Path,
    lang: str,
) -> Tuple[int, int, float]:
    instances = read_offline_instances(instances_log)
    source_lines = read_text_lines(source_file)
    if len(instances) != len(source_lines):
        raise ValueError(
            f"offline source/instance row mismatch for {instances_log}: "
            f"instances={len(instances)} source_lines={len(source_lines)}"
        )
    terms = load_streamlaal_terms(glossary_path, lang)
    correct = 0
    total = 0
    for idx, inst in enumerate(instances):
        ref = str(inst.get("reference") or "")
        pred = str(inst.get("prediction") or "")
        source_ref = source_lines[idx]
        for term_info in terms:
            target = term_info["target"]
            if source_contains(source_ref, term_info.get("en", "")) and target in ref:
                total += 1
                if target in pred:
                    correct += 1
    if total <= 0:
        raise ValueError(f"zero TERM_TOTAL for {instances_log}")
    return correct, total, correct / total


def offline_llm_row(
    *,
    dataset: str,
    method: str,
    lang: str,
    result_dir: Path,
    glossary_path: Path,
    source_file: Path,
    note: str,
) -> Dict[str, str]:
    scores_path = result_dir / "scores.tsv"
    instances_log = result_dir / "instances.log"
    scores = read_single_tsv_row(scores_path)
    correct, total, acc = score_offline_term_accuracy(
        instances_log=instances_log,
        source_file=source_file,
        glossary_path=glossary_path,
        lang=lang,
    )
    return row(
        dataset=dataset,
        method=method,
        lang=lang,
        lm=None,
        bleu=numeric(scores, "BLEU", scores_path),
        term_acc=acc,
        term_correct=correct,
        term_total=total,
        source_type="offline_llm_parsed",
        source_path=str(result_dir),
        status="offline_reference",
        note=note,
    )


def numeric(raw: Mapping[str, str], key: str, path: Path) -> float:
    value = raw.get(key, "")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric {key}={value!r} in {path}") from exc


def intish(raw: Mapping[str, str], key: str) -> int | None:
    value = raw.get(key, "")
    if value in {"", "NA", "N/A", None}:
        return None
    return int(float(value))


def eval_result_row(
    *,
    dataset: str,
    method: str,
    lang: str,
    lm: int,
    src: EvalSource,
) -> Dict[str, str]:
    raw = read_last_tsv_row(src.path)
    return row(
        dataset=dataset,
        method=method,
        lang=lang,
        lm=lm,
        bleu=numeric(raw, "BLEU", src.path),
        streamlaal=numeric(raw, "StreamLAAL", src.path),
        streamlaal_ca=numeric(raw, "StreamLAAL_CA", src.path),
        term_acc=numeric(raw, "TERM_ACC", src.path),
        term_correct=intish(raw, "TERM_CORRECT"),
        term_total=intish(raw, "TERM_TOTAL"),
        source_type=src.source_type,
        source_path=str(src.path),
        event_id=src.event_id,
        wandb_run_id=src.wandb_run_id,
        status=src.status,
        note=src.note,
    )


def source_has_five_instances(path: Path) -> bool:
    raw = read_last_tsv_row(path)
    instances = raw.get("instances_log", "")
    if not instances or instances in {"NA", "N/A"}:
        return False
    inst_path = Path(instances)
    if not inst_path.exists():
        return False
    with inst_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [line for line in f if line.strip()]
    return len(lines) == 5


def placeholder(
    *,
    dataset: str,
    method: str,
    lang: str,
    lm: int | str | None,
    status: str,
    note: str,
    source_path: str = USER_SOURCE,
) -> Dict[str, str]:
    return row(
        dataset=dataset,
        method=method,
        lang=lang,
        lm=lm,
        source_type="placeholder",
        source_path=source_path,
        status=status,
        note=note,
    )


def add_static_acl_tagged(rows: List[Dict[str, str]]) -> None:
    baseline = {
        "zh": [
            (1, 40.6663, 1181.1470, 1503.3985, 0.7431),
            (2, 45.8268, 1765.7196, 2124.4651, 0.7655),
            (3, 46.7119, 2232.6733, 2679.6834, 0.7675),
            (4, 47.3897, 2616.3493, 3187.8067, 0.7754),
        ],
        "ja": [
            (1, 22.0137, 1571.0, 2059.0, 0.6331),
            (2, 27.8786, 2300.0, 3194.0, 0.6564),
            (3, 29.3039, 2707.0, 3342.0, 0.6724),
            (4, 30.6042, 3252.0, 4090.0, 0.6751),
        ],
        "de": [
            (1, 27.4672, 1124.0, 1507.0, 0.6496),
            (2, 31.6370, 1773.0, 2273.0, 0.6744),
            (3, 31.7033, 2383.0, 3048.0, 0.6864),
            (4, 28.6382, 2719.0, 3618.0, 0.6592),
        ],
    }
    baseline_overrides: Dict[Tuple[str, int], EvalSource] = {
        ("de", 4): EvalSource(
            Path(
                "/mnt/gemini/data1/jiaxuanluo/"
                "tagged_acl_origin_norag_de_lm4_raw_rerun_"
                "20260524T160830_tagacl_origin_norag_de_lm4_raw_rerun/"
                "origin_norag/de/"
                "gigaspeech-de-s_origin-bsz4_gacl6060_tagged_gt_raw_min_norm2_"
                "cs3.84_hs0.48_lm4_k210_k110_th0p0/eval_results.tsv"
            ),
            "20260524T160830__simuleval__tagged_acl_origin_norag_de_lm4_raw_rerun",
            "3upoqej5",
            note="verified rerun replacing abnormal InfiniSST de lm4 point",
        )
    }

    for lang, _ in LANGS:
        rows.append(
            offline_llm_row(
                dataset="acl_tagged_raw",
                method="Offline ST",
                lang=lang,
                result_dir=OFFLINE_LLM_ROOT / "acl6060" / lang,
                glossary_path=ACL_TAGGED_RAW_GLOSSARY,
                source_file=ACL_SOURCE_FILE,
                note="offline full-context LLM baseline from Siqi's offline/acl6060 outputs; plotted as no-latency horizontal reference",
            )
        )
        rows.append(
            offline_llm_row(
                dataset="acl_tagged_raw",
                method="Offline + GT terms",
                lang=lang,
                result_dir=OFFLINE_LLM_ROOT / "acl6060/glossary" / lang,
                glossary_path=ACL_TAGGED_RAW_GLOSSARY,
                source_file=ACL_SOURCE_FILE,
                note="offline full-context LLM with oracle/GT terms from Siqi's offline/acl6060/glossary outputs; plotted as oracle-term horizontal reference",
            )
        )
    for lang, values in baseline.items():
        for lm, bleu, streamlaal, streamlaal_ca, term_acc in values:
            override = baseline_overrides.get((lang, lm))
            if override is not None:
                rows.append(
                    eval_result_row(
                        dataset="acl_tagged_raw",
                        method="InfiniSST",
                        lang=lang,
                        lm=lm,
                        src=override,
                    )
                )
                continue
            rows.append(
                row(
                    dataset="acl_tagged_raw",
                    method="InfiniSST",
                    lang=lang,
                    lm=lm,
                    bleu=bleu,
                    streamlaal=streamlaal,
                    streamlaal_ca=streamlaal_ca,
                    term_acc=term_acc,
                    source_type="user_supplied_reusable",
                    source_path=USER_SOURCE,
                    status="baseline_reference",
                    note="ACL tagged raw baseline supplied by user",
                )
            )


def add_acl_tagged_rasst(rows: List[Dict[str, str]]) -> None:
    root = Path("/mnt/gemini/data1/jiaxuanluo")
    sources: Dict[Tuple[str, int], EvalSource] = {
        ("zh", 1): EvalSource(
            root
            / "tagged_acl_same_lm_batch_v1_hn1024_tau078_raw_zh_lm1_max256_20260524T1442_tagacl_same_lm_batch_v1exact_hn1024_tau078_raw_zh_lm1_max256_taurus45/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078_same_lm_batch_v1_max256/zh/dtagacl_same_lm_batch_v1_hn1024_tau078_max256_lm1_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv",
            "20260524T1442__simuleval__tagged_acl_same_lm_batch_v1exact_hn1024_tau078_raw_zh_lm1_max256",
            "kolja8vr",
            note="same-lm batch max256 readout requested by user",
        ),
        ("zh", 2): EvalSource(
            root
            / "tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_20260524T0522_tagacl_newv9_hn1024_tau078_raw_zh_lm23_aries4567/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/zh/dtagacl_new_v9_hn1024_tau078_lm2_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv",
            "20260524T0522__simuleval__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_aries4567",
            "hplut7h5",
        ),
        ("zh", 3): EvalSource(
            root
            / "tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_20260524T0522_tagacl_newv9_hn1024_tau078_raw_zh_lm23_aries4567/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/zh/dtagacl_new_v9_hn1024_tau078_lm3_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv",
            "20260524T0522__simuleval__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm23_aries4567",
            "a7bqd6nu",
        ),
        ("zh", 4): EvalSource(
            root
            / "tagged_acl_new_v9_hn1024_tau078_raw_zh_lm4_20260524T0555_tagacl_newv9_hn1024_tau078_raw_zh_lm4_aries45/new_v9_termtag_delay_oldnewv3_r32a64_hn1024_tau078/zh/dtagacl_new_v9_hn1024_tau078_lm4_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv",
            "20260524T0555__simuleval__tagged_acl_new_v9_hn1024_tau078_raw_zh_lm4_aries45",
            "",
            note="eval_results exists although source manifest was not fully re-registered after completion",
        ),
        ("de", 1): EvalSource(
            root
            / "tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/de/dtagacl_bv1_mfa_np_hn1024_tau078_lm1_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/wordfix_eval/eval_results.tsv",
            "20260524T1738__simuleval__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_de_lm1to4_max80",
            "",
            note="clean MFA/source-filtered de New V9, same-lm batch max_new_tokens=80; de word-boundary fixed; lm1 rerun with audio limit 128",
        ),
        ("de", 2): EvalSource(
            root
            / "tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/de/dtagacl_bv1_mfa_np_hn1024_tau078_lm2_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/wordfix_eval/eval_results.tsv",
            "20260524T1738__simuleval__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_de_lm1to4_max80",
            "",
            note="clean MFA/source-filtered de New V9, same-lm batch max_new_tokens=80; de word-boundary fixed",
        ),
        ("de", 3): EvalSource(
            root
            / "tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/de/dtagacl_bv1_mfa_np_hn1024_tau078_lm3_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/wordfix_eval/eval_results.tsv",
            "20260524T1738__simuleval__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_de_lm1to4_max80",
            "",
            note="clean MFA/source-filtered de New V9, same-lm batch max_new_tokens=80; de word-boundary fixed",
        ),
        ("de", 4): EvalSource(
            root
            / "tagged_acl_same_lm_batch_v1_mfa_npfilter_hn1024_tau078_raw_de_lm1to4_20260524T1738_tagacl_newv9_mfa_npfilter_de_batch_mt80/new_v9_mfa_npfilter_lexexact_oldnewv3_de_r32a64_hn1024_tau078_same_lm_batch_v1/de/dtagacl_bv1_mfa_np_hn1024_tau078_lm4_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/wordfix_eval/eval_results.tsv",
            "20260524T1738__simuleval__tagged_acl_new_v9_mfa_npfilter_lexexact_hn1024_tau078_raw_de_lm1to4_max80",
            "",
            note="clean MFA/source-filtered de New V9, same-lm batch max_new_tokens=80; de word-boundary fixed",
        ),
        ("ja", 1): EvalSource(
            Path("/mnt/data1/jiaxuanluo/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_20260526T014004_tagged_acl_ja_lm1_serial_promptfix_cache30_vllmaudio128_max40lm_taurus/ja/dtagacl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_lm1_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv"),
            "20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus",
            source_type="serial_eval_results",
            note="ja NewV9 tagged-term SLM; tagged ACL raw glossary; HN1024 tau=0.78; VLLM_LIMIT_AUDIO=128; cache 30/30; max_new_tokens=40*lm; empty maps omitted; system prompt duplication fixed",
        ),
        ("ja", 2): EvalSource(
            Path("/mnt/data1/jiaxuanluo/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_20260526T014004_tagged_acl_ja_lm2_serial_promptfix_cache30_vllmaudio128_max40lm_taurus/ja/dtagacl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_lm2_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv"),
            "20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus",
            source_type="serial_eval_results",
            note="ja NewV9 tagged-term SLM; tagged ACL raw glossary; HN1024 tau=0.78; VLLM_LIMIT_AUDIO=128; cache 30/30; max_new_tokens=40*lm; empty maps omitted; system prompt duplication fixed",
        ),
        ("ja", 3): EvalSource(
            Path("/mnt/data1/jiaxuanluo/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_20260526T014004_tagged_acl_ja_lm3_serial_promptfix_cache30_vllmaudio128_max40lm_taurus/ja/dtagacl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_lm3_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv"),
            "20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus",
            source_type="serial_eval_results",
            note="ja NewV9 tagged-term SLM; tagged ACL raw glossary; HN1024 tau=0.78; VLLM_LIMIT_AUDIO=128; cache 30/30; max_new_tokens=40*lm; empty maps omitted; system prompt duplication fixed",
        ),
        ("ja", 4): EvalSource(
            Path("/mnt/data1/jiaxuanluo/tagged_acl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_20260526T014004_tagged_acl_ja_lm4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus/ja/dtagacl_ja_serial_promptfix_cache30_vllmaudio128_max40lm_lm4_k10_th0.78_gacl6060_tagged_gt_raw_min_norm2/eval_results.tsv"),
            "20260526T013751__simuleval__tagged_acl_ja_lm1to4_serial_promptfix_cache30_vllmaudio128_max40lm_taurus",
            source_type="serial_eval_results",
            note="ja NewV9 tagged-term SLM; tagged ACL raw glossary; HN1024 tau=0.78; VLLM_LIMIT_AUDIO=128; cache 30/30; max_new_tokens=40*lm; empty maps omitted; system prompt duplication fixed",
        ),
    }
    for (lang, lm), src in sorted(sources.items()):
        rows.append(eval_result_row(dataset="acl_tagged_raw", method="RASST", lang=lang, lm=lm, src=src))


def add_acl_paper_extracted(rows: List[Dict[str, str]]) -> None:
    offline = {
        "zh": (49.6625, 0.7941, 297, 374),
        "ja": (32.9328, 0.7623, 202, 265),
        "de": (35.7641, 0.7261, 228, 314),
    }
    baseline = {
        "zh": [
            (1, 40.86, 1142, 1576, 0.7353),
            (2, 45.66, 1789, 2289, 0.7299),
            (3, 47.50, 2216, 2886, 0.7540),
            (4, 47.73, 2583, 3304, 0.7754),
        ],
        "ja": [
            (1, 21.35, 1554, 2148, 0.6679),
            (2, 28.04, 2287, 2893, 0.6491),
            (3, 28.94, 2693, 3445, 0.7057),
            (4, 30.58, 3330, 4243, 0.7472),
        ],
        "de": [
            (1, 27.29, 1132, 1645, 0.5732),
            (2, 30.89, 1829, 2464, 0.5701),
            (3, 31.24, 2373, 3275, 0.6815),
            (4, 31.42, 2826, 3932, 0.6561),
        ],
    }
    rasst = {
        "zh": [
            (1, 43.30, 1219, 1724, 0.8075),
            (2, 47.50, 1792, 2432, 0.8930),
            (3, 49.05, 2280, 3052, 0.8556),
            (4, 49.07, 2703, 3676, 0.8930),
        ],
        "ja": [
            (1, 19.10, 1371, 2176, 0.7208),
            (2, 27.38, 2045, 2874, 0.7660),
            (3, 29.31, 2670, 3697, 0.8075),
            (4, 30.85, 3050, 4417, 0.8226),
        ],
        "de": [
            (1, 28.29, 1108, 1743, 0.7038),
            (2, 31.84, 1704, 3911, 0.7994),
            (3, 30.10, 2158, 3251, 0.7134),
            (4, 28.87, 2591, 3921, 0.7580),
        ],
    }
    for lang, (bleu, term_acc, correct, total) in offline.items():
        rows.append(
            row(
                dataset="acl_paper_extracted",
                method="Offline ST",
                lang=lang,
                lm=None,
                bleu=bleu,
                term_acc=term_acc,
                term_correct=correct,
                term_total=total,
                source_type="user_supplied_reusable",
                source_path=USER_SOURCE,
                status="offline_reference",
                note="paper-extracted offline row preserved in TSV only",
            )
        )
    for method, table in (("InfiniSST", baseline), ("RASST", rasst)):
        for lang, values in table.items():
            for lm, bleu, streamlaal, streamlaal_ca, term_acc in values:
                rows.append(
                    row(
                        dataset="acl_paper_extracted",
                        method=method,
                        lang=lang,
                        lm=lm,
                        bleu=bleu,
                        streamlaal=streamlaal,
                        streamlaal_ca=streamlaal_ca,
                        term_acc=term_acc,
                        source_type="user_supplied_reusable",
                        source_path=USER_SOURCE,
                        status="reference_user_supplied",
                        note="preserved for provenance; not used in new figures",
                    )
                )


def add_medicine_offline(rows: List[Dict[str, str]]) -> None:
    for lang, _ in LANGS:
        rows.append(
            offline_llm_row(
                dataset="medicine_hardraw",
                method="Offline ST",
                lang=lang,
                result_dir=OFFLINE_LLM_ROOT / "eso/baseline" / lang,
                glossary_path=MEDICINE_HARDRAW_GLOSSARY,
                source_file=MEDICINE_SOURCE_FILE,
                note="offline full-context LLM baseline from Siqi's offline/eso/baseline outputs; plotted as no-latency horizontal reference",
            )
        )
        rows.append(
            offline_llm_row(
                dataset="medicine_hardraw",
                method="Offline + GT terms",
                lang=lang,
                result_dir=OFFLINE_LLM_ROOT / "eso/glossary" / lang,
                glossary_path=MEDICINE_HARDRAW_GLOSSARY,
                source_file=MEDICINE_SOURCE_FILE,
                note="offline full-context LLM with oracle/GT terms from Siqi's offline/eso/glossary outputs; plotted as oracle-term horizontal reference",
            )
        )


def add_medicine_baseline(rows: List[Dict[str, str]]) -> None:
    root = Path("/mnt/gemini/data1/jiaxuanluo")
    sources: Dict[Tuple[str, int], Tuple[Path, str, str]] = {
        ("zh", 1): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm1_aries01/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs0.96_hs0.48_lm1_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0610__simuleval__medicine_norag_abbrev_restored_zh_lm13_lm4no605000_aries",
            "",
        ),
        ("zh", 2): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260522/zh/gigaspeech-zh-s_origin-bsz4_gmedicine_gt571_abbrev_restored__medicine5_cs1.92_hs0.48_lm2_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0717__analysis__medicine_zh_hard_manual_lm2_lm4_605000_posteval",
            "",
        ),
        ("zh", 3): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm3_aries67/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs2.88_hs0.48_lm3_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0610__simuleval__medicine_norag_abbrev_restored_zh_lm13_lm4no605000_aries",
            "",
        ),
        ("zh", 4): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_zh_lm4_with605000_from_taurus_orig80/zh/gigaspeech-zh-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs3.84_hs0.48_lm4_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0717__analysis__medicine_zh_hard_manual_lm2_lm4_605000_posteval",
            "",
        ),
        ("de", 1): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm1_aries23/de/gigaspeech-de-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs0.96_hs0.48_lm1_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0305__simuleval__medicine_norag_abbrev_restored_de_lm1234_aries01234567",
            "",
        ),
        ("de", 2): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm2_aries45/de/gigaspeech-de-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs1.92_hs0.48_lm2_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0305__simuleval__medicine_norag_abbrev_restored_de_lm1234_aries01234567",
            "",
        ),
        ("de", 3): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_de_rerun_shorttmp_lm3_aries67/de/gigaspeech-de-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs2.88_hs0.48_lm3_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0305__simuleval__medicine_norag_abbrev_restored_de_lm1234_aries01234567",
            "",
        ),
        ("de", 4): (
            root
            / "medicine_norag_baseline_de_lm4_batch_max80_aries67_20260524T175015_medicine_norag_de_lm4_batch_max80_aries67/batch_eval/de/dmedicine_norag_baseline_batch_max80_lm4_k0_th0.0_ghard_llm_manual_check/eval_results_streamlaal_term.hard_llm_manual_check.boundaryfix.tsv",
            "20260524T175015__simuleval__medicine_norag_de_lm4_baseline_batch_max80_aries67",
            "",
        ),
        ("ja", 1): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_ja_lm1_aries01/ja/gigaspeech-ja-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs0.96_hs0.48_lm1_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T1731__analysis__medicine_ja_lm1_hard_manual_posteval",
            "",
        ),
        ("ja", 2): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260523_ja_lm2_aries45/ja/gigaspeech-ja-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs1.92_hs0.48_lm2_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260523T2342__simuleval__medicine_norag_abbrev_restored_ja_lm2_aries45",
            "",
        ),
        ("ja", 3): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_ja_lm3_aries23/ja/gigaspeech-ja-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs2.88_hs0.48_lm3_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0130__simuleval__medicine_norag_abbrev_restored_ja_lm134_aries012345",
            "",
        ),
        ("ja", 4): (
            root
            / "medicine_norag_baseline_abbrev_restored_batched_20260524_ja_lm4_aries45/ja/gigaspeech-ja-s_origin-bsz4_gstrict_fixed_medicine_glossary_abbrev_restored__medicine5_cs3.84_hs0.48_lm4_k210_k110_th0p0/eval_results_streamlaal_term.hard_llm_manual_check.tsv",
            "20260524T0130__simuleval__medicine_norag_abbrev_restored_ja_lm134_aries012345",
            "",
        ),
    }
    failure_notes: Dict[Tuple[str, int], str] = {}
    for lang, _ in LANGS:
        for lm in LMS:
            item = sources.get((lang, lm))
            if not item:
                rows.append(
                    placeholder(
                        dataset="medicine_hardraw",
                        method="InfiniSST",
                        lang=lang,
                        lm=lm,
                        status="placeholder_missing_baseline",
                        note="no hard-manual five-sample baseline TSV found",
                    )
                )
                continue
            path, event_id, wandb_run_id = item
            if not path.exists() or not source_has_five_instances(path):
                failed_note = failure_notes.get((lang, lm))
                rows.append(
                    placeholder(
                        dataset="medicine_hardraw",
                        method="InfiniSST",
                        lang=lang,
                        lm=lm,
                        status="placeholder_failed_baseline" if failed_note else "placeholder_missing_baseline",
                        note=failed_note or "baseline hard-manual TSV missing or did not pass five-instance check",
                        source_path=str(path),
                    )
                )
                continue
            rows.append(
                eval_result_row(
                    dataset="medicine_hardraw",
                    method="InfiniSST",
                    lang=lang,
                    lm=lm,
                    src=EvalSource(
                        path,
                        event_id,
                        wandb_run_id,
                        source_type="verified_hard_manual_streamlaal",
                        status="verified",
                        note=(
                            "baseline no-RAG hard-manual StreamLAAL/TERM post-eval; "
                            "de lm4 uses boundaryfix TSV rebuilt from runtime chunks"
                            if (lang, lm) == ("de", 4)
                            else "baseline no-RAG hard-manual StreamLAAL/TERM post-eval"
                        ),
                    ),
                )
            )


def add_medicine_rasst(rows: List[Dict[str, str]]) -> None:
    root = Path("/mnt/gemini/data1/jiaxuanluo")
    zh_base = root / "medicine_hardraw_hn1024_tau078_new_v9_batch_20260524T0242/zh"
    for lm in LMS:
        event_id = "20260524T0242__simuleval__medicine_hardraw_hn1024_tau078_new_v9_batch"
        note = "clean zh New V9 hardraw RASST"
        search_base = zh_base
        paths = sorted(search_base.glob(f"*raw_lm{lm}_*/eval_results.tsv"))
        if len(paths) != 1:
            raise ValueError(f"expected one zh medicine RASST lm{lm} file, found {paths}")
        rows.append(
            eval_result_row(
                dataset="medicine_hardraw",
                method="RASST",
                lang="zh",
                lm=lm,
                src=EvalSource(
                    paths[0],
                    event_id,
                    "",
                    source_type="verified_eval_results",
                    status="verified",
                    note=note,
                ),
            )
        )

    de12_base = (
        root
        / "de_cap16_denoise_medicine_acl_batch_taurus_20260525T165125_de_cap16_denoise_med_acl_batch_taurus/"
        "medicine_hardraw_de_cap16_denoise_ttag_hn1024_tau078_batch_chunks30/de"
    )
    de34_base = (
        root
        / "medicine_hardraw_de_cap16_denoise_lm34_batch_taurus03_20260525T170456_medicine_de_cap16_denoise_lm34_batch_taurus03/de"
    )
    de_sources = {
        1: (
            de12_base,
            "*lm1_k10_th0.78_*/eval_results.tsv",
            "20260525T165125__simuleval__de_cap16_denoise_medicine_acl_batch_taurus",
        ),
        2: (
            de12_base,
            "*lm2_k10_th0.78_*/eval_results.tsv",
            "20260525T165125__simuleval__de_cap16_denoise_medicine_acl_batch_taurus",
        ),
        3: (
            de34_base,
            "*lm34_taurus03_lm3_k10_th0.78_*/eval_results.tsv",
            "20260525T170456__simuleval__medicine_de_cap16_denoise_lm34_batch_taurus03",
        ),
        4: (
            de34_base,
            "*lm34_taurus03_lm4_k10_th0.78_*/eval_results.tsv",
            "20260525T170456__simuleval__medicine_de_cap16_denoise_lm34_batch_taurus03",
        ),
    }
    for lm, (base, pattern, event_id) in sorted(de_sources.items()):
        paths = sorted(base.glob(pattern))
        if len(paths) != 1:
            raise ValueError(f"expected one de medicine RASST lm{lm} file, found {paths}")
        rows.append(
            eval_result_row(
                dataset="medicine_hardraw",
                method="RASST",
                lang="de",
                lm=lm,
                src=EvalSource(
                    paths[0],
                    event_id,
                    "",
                    source_type="verified_eval_results",
                    status="verified",
                    note=(
                        "cap16-denoise tagged-term SLM, medicine hardraw glossary, "
                        "HN1024 tau=0.78, empty maps omitted, cache chunks30; batch eval"
                    ),
                ),
            )
        )

    ja_base = root / "medicine_hardraw_20260525T184043_ja_med_cap16den_lm1234_taurus/ja"
    for lm in LMS:
        paths = sorted(ja_base.glob(f"*lm{lm}_k10_th0.78_*/eval_results.tsv"))
        if len(paths) != 1:
            raise ValueError(f"expected one ja medicine RASST lm{lm} file, found {paths}")
        rows.append(
            eval_result_row(
                dataset="medicine_hardraw",
                method="RASST",
                lang="ja",
                lm=lm,
                src=EvalSource(
                    paths[0],
                    "20260525T1840__simuleval__medicine_ja_cap16_denoise_lm1234_batch_taurus",
                    "",
                    source_type="verified_eval_results",
                    status="verified",
                    note=(
                        "cap16-denoise tagged-term SLM, medicine hardraw glossary, "
                        "HN1024 tau=0.78, empty maps omitted, cache chunks30, "
                        "max_new_tokens=lm*40; batch eval"
                    ),
                ),
            )
        )


def build_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    add_static_acl_tagged(rows)
    add_acl_tagged_rasst(rows)
    add_acl_paper_extracted(rows)
    add_medicine_offline(rows)
    add_medicine_baseline(rows)
    add_medicine_rasst(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    seen: MutableMapping[Tuple[str, str, str, str], Mapping[str, str]] = {}
    allowed_statuses = {
        "offline_reference",
        "baseline_reference",
        "reference_user_supplied",
        "verified",
        "dirty_untrusted",
        "placeholder_missing_offline",
        "placeholder_missing_baseline",
        "placeholder_failed_baseline",
        "placeholder_missing_rasst",
    }
    for r in rows:
        missing = [field for field in FIELDS if field not in r]
        if missing:
            raise ValueError(f"missing fields {missing} in {r}")
        key = (r["dataset"], r["method"], r["lang"], r["lm"])
        if key in seen:
            raise ValueError(f"duplicate row key {key}")
        seen[key] = r
        if r["status"] not in allowed_statuses:
            raise ValueError(f"unsupported status {r['status']} in {r}")
        if r["status"] == "dirty_untrusted" and not (
            r["dataset"] == "medicine_hardraw" and r["method"] == "RASST" and r["lang"] == "de"
        ):
            raise ValueError(f"dirty_untrusted only allowed for medicine RASST de: {r}")
        for field in ("BLEU", "StreamLAAL", "StreamLAAL_CA", "TERM_ACC", "TERM_CORRECT", "TERM_TOTAL"):
            value = r[field]
            if value in {"NA", ""}:
                if r["status"].startswith("placeholder"):
                    continue
                if r["method"] in OFFLINE_METHODS and field in {"StreamLAAL", "StreamLAAL_CA"}:
                    continue
                if field in {"TERM_CORRECT", "TERM_TOTAL"} and r["source_type"] == "user_supplied_reusable":
                    continue
                raise ValueError(f"unexpected NA in {field}: {r}")
            try:
                float(value)
            except ValueError as exc:
                raise ValueError(f"non-numeric {field}={value!r}: {r}") from exc


def write_tsv(rows: Sequence[Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def finite(value: str) -> float | None:
    if value in {"", "NA"}:
        return None
    return float(value)


METHOD_STYLES = {
    "Offline ST": {"color": "#3a923a", "linestyle": "--", "linewidth": 1.9},
    "Offline + GT terms": {"color": "#805ad5", "linestyle": "-.", "linewidth": 1.9},
    "InfiniSST": {
        "color": "#2b6cb0",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 1.9,
        "markersize": 6.2,
    },
    "RASST": {
        "color": "#d62728",
        "marker": "*",
        "linestyle": "-",
        "linewidth": 2.0,
        "markersize": 9.0,
    },
}
OFFLINE_METHODS = ("Offline ST", "Offline + GT terms")
PLOT_METHODS = (*OFFLINE_METHODS, "InfiniSST", "RASST")


def red_status_notes(lang_rows: Sequence[Mapping[str, str]]) -> List[str]:
    """Summarize missing or dirty rows for compact in-figure red annotations."""
    notes: List[str] = []
    by_method: MutableMapping[str, List[Mapping[str, str]]] = {}
    for r in lang_rows:
        by_method.setdefault(r["method"], []).append(r)

    for method in PLOT_METHODS:
        rows = by_method.get(method, [])
        if not rows:
            continue
        placeholders = [
            r
            for r in rows
            if r["status"].startswith("placeholder")
            and r["lm"] != "NA"
        ]
        offline_missing = any(
            r["method"] == "Offline ST" and r["status"].startswith("placeholder")
            for r in rows
        )
        dirty = [r for r in rows if r["status"] == "dirty_untrusted"]

        if offline_missing:
            notes.append(f"{method}: unavailable")
        failed = [r for r in placeholders if "failed" in r["status"] or "failed" in r["note"].lower()]
        if placeholders and len(placeholders) == len(rows) and not failed:
            notes.append(f"{method}: unavailable")
        elif placeholders:
            unavailable = [r for r in placeholders if r not in failed]
            if unavailable:
                lms = ",".join(r["lm"] for r in sorted(unavailable, key=lambda item: int(item["lm"])))
                notes.append(f"{method}: lm{lms} unavailable")
            if failed:
                lms = ",".join(r["lm"] for r in sorted(failed, key=lambda item: int(item["lm"])))
                notes.append(f"{method}: lm{lms} failed")
        if dirty:
            notes.append(f"{method}: dirty/provisional")

    return notes


def plot_dataset(rows: Sequence[Mapping[str, str]], dataset: str, output_prefix: Path) -> None:
    data = [r for r in rows if r["dataset"] == dataset]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.9,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.8))
    handles: List[object] = []
    labels: List[str] = []

    for col, (lang, title) in enumerate(LANGS):
        lang_rows = [r for r in data if r["lang"] == lang]
        for row_idx, metric in enumerate(("TERM_ACC", "BLEU")):
            ax = axes[row_idx][col]
            x_values: List[float] = []
            y_values: List[float] = []
            for method in PLOT_METHODS:
                method_rows = [r for r in lang_rows if r["method"] == method]
                if method in OFFLINE_METHODS:
                    offline = next((r for r in method_rows if finite(r[metric]) is not None), None)
                    if offline:
                        y = finite(offline[metric])
                        assert y is not None
                        line = ax.axhline(y * (100.0 if metric == "TERM_ACC" else 1.0), label=method, **METHOD_STYLES[method])
                        y_values.append(y * (100.0 if metric == "TERM_ACC" else 1.0))
                        if col == 0 and row_idx == 0:
                            handles.append(line)
                            labels.append(method)
                    continue
                points = []
                for r in sorted(method_rows, key=lambda item: int(item["lm"]) if item["lm"].isdigit() else 99):
                    x = finite(r["StreamLAAL"])
                    y = finite(r[metric])
                    if x is None or y is None:
                        continue
                    points.append((x, y * (100.0 if metric == "TERM_ACC" else 1.0), r["status"]))
                if not points:
                    continue
                style = dict(METHOD_STYLES[method])
                if any(status == "dirty_untrusted" for _, _, status in points):
                    style.update({"linestyle": "--", "markerfacecolor": "white", "markeredgewidth": 1.2})
                line = ax.plot([p[0] for p in points], [p[1] for p in points], label=method, **style)[0]
                x_values.extend(p[0] for p in points)
                y_values.extend(p[1] for p in points)
                if col == 0 and row_idx == 0 and method not in labels:
                    handles.append(line)
                    labels.append(method)

            if x_values:
                x_low = min(x_values)
                x_high = max(x_values)
                pad = max((x_high - x_low) * 0.10, 90.0)
                ax.set_xlim(x_low - pad, x_high + pad)
            if y_values:
                y_low = min(y_values)
                y_high = max(y_values)
                pad = max((y_high - y_low) * 0.10, 1.2 if metric == "BLEU" else 1.8)
                ax.set_ylim(y_low - pad, y_high + pad)
            ax.grid(True, linestyle=":", linewidth=0.55, alpha=0.65)
            if row_idx == 0:
                ax.set_title(title, fontweight="bold")
            else:
                ax.set_xlabel("StreamLAAL (ms)")
            if col == 0:
                ax.set_ylabel("Terminology\nAccuracy (%)" if metric == "TERM_ACC" else "BLEU Score")

            if row_idx == 0:
                notes = red_status_notes(lang_rows)
                if notes:
                    ax.text(
                        0.02,
                        0.97,
                        "\n".join(notes),
                        transform=ax.transAxes,
                        va="top",
                        ha="left",
                        color="#c00000",
                        fontsize=7.8,
                        linespacing=1.15,
                        bbox={
                            "facecolor": "white",
                            "edgecolor": "#c00000",
                            "linewidth": 0.55,
                            "alpha": 0.88,
                            "boxstyle": "round,pad=0.22",
                        },
                    )

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(1, len(labels)),
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
            columnspacing=1.5,
            handlelength=2.0,
        )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0), w_pad=1.2, h_pad=1.6)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-tsv", type=Path, default=REPORT_DIR / "20260524_main_result_data.tsv")
    parser.add_argument("--figure-dir", type=Path, default=PAPER_FIG_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows()
    validate_rows(rows)
    write_tsv(rows, args.output_tsv)
    loaded = load_rows(args.output_tsv)
    validate_rows(loaded)
    plot_dataset(loaded, "acl_tagged_raw", args.figure_dir / "new_main_result_tagged")
    plot_dataset(loaded, "medicine_hardraw", args.figure_dir / "medicine_main_result")
    print(f"Wrote {args.output_tsv}")
    print(f"Wrote {args.figure_dir / 'new_main_result_tagged.pdf'}")
    print(f"Wrote {args.figure_dir / 'medicine_main_result.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
