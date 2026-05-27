#!/usr/bin/env python3
"""Compute sentence-level terminology adoption from SimulEval outputs.

``TERM_ADOPTION`` is the gold-source metric: source/reference terms are the
denominator even if the retriever never put them in the prompt.  When a runtime
log and audio yaml are provided, ``REAL_TERM_ADOPT`` narrows that denominator to
gold terms that were actually present in the sentence-aligned term_map.

``TERM_SENT_FCR`` is a sentence-level true false-copy rate.  In the fixed
glossary policy, its denominator is the number of sentences containing at least
one glossary entry whose source term is absent from the source sentence and
whose target translation is absent from the reference sentence.  Its numerator
is the number of those sentences where the hypothesis copies at least one such
unsupported translation.  This is the policy to use for fixed-domain oracle
readouts, where all term metrics must be computed from the same strict glossary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


TARGET_LANG_DEFAULT = "zh"


@dataclass(frozen=True)
class GlossaryTerm:
    term: str
    translation: str

    @property
    def match_key(self) -> Tuple[str, str]:
        return (_normalise_space(self.term).casefold(), _normalise_space(self.translation))


@dataclass(frozen=True)
class RuntimeTermMapCall:
    start_sec: float
    end_sec: float
    references: List[Dict[str, Any]]


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _load_glossary_terms(path: Path, target_lang: str) -> List[GlossaryTerm]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_entries: Iterable[Any]
    if isinstance(data, dict):
        raw_entries = data.values()
    elif isinstance(data, list):
        raw_entries = data
    else:
        raise ValueError(f"Unsupported glossary format: {path}")

    terms: List[GlossaryTerm] = []
    seen = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        term = _normalise_space(entry.get("term") or entry.get("source") or "")
        translations = entry.get("target_translations")
        translation = ""
        if isinstance(translations, dict):
            translation = _normalise_space(translations.get(target_lang) or "")
        if not translation:
            translation = _normalise_space(
                entry.get("translation")
                or entry.get("target_translation")
                or entry.get(target_lang)
                or ""
            )
        if not term or not translation:
            continue
        key = (term.casefold(), translation)
        if key in seen:
            continue
        seen.add(key)
        terms.append(GlossaryTerm(term=term, translation=translation))
    return terms


def _source_contains(source_text: str, term: str) -> bool:
    source_norm = _normalise_space(source_text).casefold()
    term_norm = _normalise_space(term).casefold()
    if not source_norm or not term_norm:
        return False
    if re.fullmatch(r"[a-z0-9][a-z0-9 ._+/#-]*", term_norm):
        pattern = r"(?<![a-z0-9])" + re.escape(term_norm) + r"(?![a-z0-9])"
        return re.search(pattern, source_norm) is not None
    return term_norm in source_norm


def _text_contains(text: str, needle: str) -> bool:
    text_norm = _normalise_space(text)
    needle_norm = _normalise_space(needle)
    return bool(needle_norm) and needle_norm in text_norm


def _load_audio_intervals(audio_yaml: Path) -> List[Tuple[float, float]]:
    with audio_yaml.open("r", encoding="utf-8") as f:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for --audio-yaml real_term_adopt") from exc
        entries = yaml.safe_load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Invalid audio yaml format (expected list): {audio_yaml}")

    intervals: List[Tuple[float, float]] = []
    cursor = 0.0
    for item in entries:
        if not isinstance(item, dict):
            continue
        duration = float(item.get("duration") or 0.0)
        start = float(item.get("offset", cursor))
        end = start + duration
        intervals.append((start, end))
        cursor = end
    return intervals


def _load_audio_sentence_metadata(audio_yaml: Path) -> List[Tuple[str, float, float]]:
    with audio_yaml.open("r", encoding="utf-8") as f:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for --audio-yaml real_term_adopt") from exc
        entries = yaml.safe_load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Invalid audio yaml format (expected list): {audio_yaml}")

    out: List[Tuple[str, float, float]] = []
    cursor_by_wav: Dict[str, float] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        wav = str(item.get("wav") or "")
        paper_id = Path(wav).stem
        duration = float(item.get("duration") or 0.0)
        default_start = cursor_by_wav.get(paper_id, 0.0)
        start = float(item.get("offset", default_start))
        end = start + duration
        out.append((paper_id, start, end))
        cursor_by_wav[paper_id] = end
    return out


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and a_end > b_start


def _ref_match_key(ref: Dict[str, Any]) -> Tuple[str, str]:
    term = _normalise_space(ref.get("term") or ref.get("key") or "")
    translation = _normalise_space(ref.get("translation") or "")
    return (term.casefold(), translation)


def _runtime_records_to_sentence_term_map_keys(
    records: Sequence[Dict[str, Any]],
    sentence_intervals: Sequence[Tuple[float, float]],
) -> List[Set[Tuple[str, str]]]:
    """Align vLLM prompt term_maps to source sentence intervals.

    Prefer timeline-aware runtime records with explicit ``current_start_sec`` /
    ``current_end_sec``.  For older final RAG records, reconstruct the current
    vLLM window from successive ``rag_audio_duration`` values.
    """
    timeline_by_segment: Dict[int, Tuple[float, float]] = {}
    final_records: List[Dict[str, Any]] = []
    llm_input_records: List[Dict[str, Any]] = []

    for rec in records:
        rec_type = rec.get("type")
        if rec_type == "rag_window" and rec.get("trigger") == "vllm_timeline":
            try:
                seg = int(rec.get("segment_idx", -1))
                start = float(rec.get("current_start_sec"))
                end = float(rec.get("current_end_sec"))
            except (TypeError, ValueError):
                continue
            if seg >= 0 and end > start:
                timeline_by_segment[seg] = (start, end)
        elif rec_type == "rag":
            final_records.append(rec)
        elif rec_type == "llm_input":
            llm_input_records.append(rec)

    refs_by_segment: Dict[int, List[Dict[str, Any]]] = {}
    for rec in final_records + llm_input_records:
        refs = [r for r in (rec.get("references") or []) if isinstance(r, dict)]
        if not refs:
            continue
        try:
            seg = int(rec.get("segment_idx", -1))
        except (TypeError, ValueError):
            continue
        if seg < 0:
            continue
        # `llm_input` is what the model saw; let it overwrite same-segment
        # `rag` records when both are present.
        refs_by_segment[seg] = refs

    calls: List[RuntimeTermMapCall] = []
    previous_end = 0.0
    for rec in final_records:
        try:
            seg = int(rec.get("segment_idx", -1))
        except (TypeError, ValueError):
            continue
        refs = refs_by_segment.get(seg)
        if not refs:
            continue
        if seg in timeline_by_segment:
            start, end = timeline_by_segment[seg]
        else:
            try:
                end = float(rec.get("rag_audio_duration") or 0.0)
            except (TypeError, ValueError):
                continue
            if end < previous_end or seg == 0:
                start = 0.0
                previous_end = end
            else:
                start = previous_end
                previous_end = end
        if end > start:
            calls.append(RuntimeTermMapCall(start_sec=start, end_sec=end, references=refs))

    # Some logs may only have llm_input records.  Fall back to segment-sized
    # sentence spans so the metric remains available for one-call-per-sentence
    # logs, but avoid inventing timing when there is no sentence interval.
    if not calls and llm_input_records and sentence_intervals:
        for rec in llm_input_records:
            try:
                seg = int(rec.get("segment_idx", -1))
            except (TypeError, ValueError):
                continue
            refs = refs_by_segment.get(seg)
            if not refs or seg < 0 or seg >= len(sentence_intervals):
                continue
            start, end = sentence_intervals[seg]
            calls.append(RuntimeTermMapCall(start_sec=start, end_sec=end, references=refs))

    out: List[Set[Tuple[str, str]]] = [set() for _ in sentence_intervals]
    for call in calls:
        call_keys = {_ref_match_key(ref) for ref in call.references if _ref_match_key(ref)[1]}
        if not call_keys:
            continue
        for i, (sent_start, sent_end) in enumerate(sentence_intervals):
            if _overlaps(call.start_sec, call.end_sec, sent_start, sent_end):
                out[i].update(call_keys)
    return out


def _load_sentence_term_map_keys(
    runtime_log: Path,
    sentence_intervals: Sequence[Tuple[float, float]],
) -> List[Set[Tuple[str, str]]]:
    return _runtime_records_to_sentence_term_map_keys(
        list(_iter_jsonl(runtime_log)),
        sentence_intervals,
    )


def _split_runtime_records_by_instance(runtime_log: Path) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    seen_positive_segment = False
    for rec in _iter_jsonl(runtime_log):
        rec_type = rec.get("type")
        seg: Optional[int] = None
        if rec_type in {"rag_window", "rag", "llm_input", "llm_output"}:
            try:
                seg = int(rec.get("segment_idx", -1))
            except (TypeError, ValueError):
                seg = None
        if (
            rec_type == "rag_window"
            and seg == 0
            and current
            and seen_positive_segment
        ):
            groups.append(current)
            current = []
            seen_positive_segment = False
        current.append(rec)
        if seg is not None and seg > 0:
            seen_positive_segment = True
    if current:
        groups.append(current)
    return groups


def _instance_paper_id(inst: Dict[str, Any]) -> Optional[str]:
    source = inst.get("source")
    candidates: List[str] = []
    if isinstance(source, list):
        candidates.extend(str(x) for x in source)
    elif isinstance(source, str):
        candidates.append(source)
    for item in candidates:
        if item.endswith(".wav"):
            return Path(item).stem
    return None


def _load_explicit_sentence_term_map_keys(
    sentence_term_map: Path,
    expected_sentences: int,
) -> List[Set[Tuple[str, str]]]:
    """Load an explicit sentence-indexed oracle/retriever term_map file."""
    data = json.loads(sentence_term_map.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"sentence term_map must be a JSON list: {sentence_term_map}")
    if len(data) != expected_sentences:
        raise ValueError(
            f"sentence term_map length {len(data)} != source sentence count {expected_sentences}: "
            f"{sentence_term_map}"
        )
    out: List[Set[Tuple[str, str]]] = []
    for row_idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"sentence term_map row {row_idx} is not an object: {sentence_term_map}")
        refs = row.get("references") or []
        if not isinstance(refs, list):
            raise ValueError(f"sentence term_map row {row_idx} references is not a list: {sentence_term_map}")
        out.append({_ref_match_key(ref) for ref in refs if isinstance(ref, dict) and _ref_match_key(ref)[1]})
    return out


def _can_count_false_copy_translation(translation: str) -> bool:
    """Avoid counting ambiguous one-character Chinese substrings as false copies."""
    needle = _normalise_space(translation)
    if not needle:
        return False
    if len(needle) == 1 and re.search(r"[\u3400-\u9fff]", needle):
        return False
    return True


def _mwer_command() -> str:
    cmd = shutil.which("mwerSegmenter")
    if cmd:
        return cmd
    root = os.environ.get("MWERSEGMENTER_ROOT", "").strip()
    if root:
        candidate = Path(root) / "mwerSegmenter"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("mwerSegmenter not found in PATH and MWERSEGMENTER_ROOT is not set")


def _segment_prediction_by_references(
    prediction: str,
    reference_sentences: List[str],
    latency_unit: str,
) -> List[str]:
    """Mirror stream_laal_term.py segmentation for sentence-level term metrics."""
    command = _mwer_command()
    character_level = latency_unit == "char"
    pred_text = str(prediction or "")
    refs = [str(x or "") for x in reference_sentences]
    if character_level:
        pred_text = " ".join(pred_text)
        refs = [" ".join(x) for x in refs]

    tmp_dir = tempfile.mkdtemp()
    pred_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    ref_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    segments_path = Path(tmp_dir) / "__segments"
    try:
        pred_file.write(pred_text)
        pred_file.flush()
        ref_file.writelines(ref + "\n" for ref in refs)
        ref_file.flush()
        subprocess.run(
            [command, "-mref", ref_file.name, "-hypfile", pred_file.name, "-usecase", "1"],
            cwd=tmp_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        segments = segments_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if character_level:
            segments = [re.sub(r"(.)\s", r"\1", line).strip() for line in segments]
        else:
            segments = [line.strip() for line in segments]
        if len(segments) != len(reference_sentences):
            raise RuntimeError(
                f"mwerSegmenter returned {len(segments)} segments for {len(reference_sentences)} references"
            )
        return segments
    finally:
        pred_file.close()
        ref_file.close()
        for p in (Path(pred_file.name), Path(ref_file.name), segments_path):
            p.unlink(missing_ok=True)
        Path(tmp_dir).rmdir()


def compute_sentence_adoption(
    instances_log: Path,
    source_file: Path,
    reference_file: Path,
    glossary_path: Path,
    target_lang: str,
    latency_unit: str,
    runtime_log: Optional[Path] = None,
    audio_yaml: Optional[Path] = None,
    sentence_term_map: Optional[Path] = None,
    fcr_policy: str = "term_map_if_available",
) -> Dict[str, Any]:
    supported_fcr_policies = {
        "term_map_if_available",
        "term_map_source_ref_negative_sentence",
        "source_ref_negative_sentence",
    }
    if fcr_policy not in supported_fcr_policies:
        raise ValueError(f"Unsupported fcr_policy={fcr_policy}")

    instances = list(_iter_jsonl(instances_log))
    sources = _read_lines(source_file)
    refs = _read_lines(reference_file)
    terms = _load_glossary_terms(glossary_path, target_lang)
    sentence_term_map_keys: Optional[List[Set[Tuple[str, str]]]] = None
    real_metric_note = ""
    multi_instance_sentence_groups: Optional[Dict[int, List[Tuple[int, str, str, str]]]] = None
    can_align_runtime = len(instances) == 1 or len(instances) == len(sources)
    if sentence_term_map is not None and sentence_term_map.is_file():
        sentence_term_map_keys = _load_explicit_sentence_term_map_keys(sentence_term_map, len(sources))
        real_metric_note = f"explicit_sentence_term_map:{sentence_term_map}"
    elif (
        runtime_log is not None
        and audio_yaml is not None
        and runtime_log.is_file()
        and audio_yaml.is_file()
        and can_align_runtime
    ):
        intervals = _load_audio_intervals(audio_yaml)
        if len(intervals) == len(sources):
            sentence_term_map_keys = _load_sentence_term_map_keys(runtime_log, intervals)
        else:
            real_metric_note = (
                f"skipped: audio intervals {len(intervals)} != source sentences {len(sources)}"
            )
    elif (
        runtime_log is not None
        and audio_yaml is not None
        and runtime_log.is_file()
        and audio_yaml.is_file()
        and len(instances) > 1
        and len(instances) < len(sources)
    ):
        metadata = _load_audio_sentence_metadata(audio_yaml)
        if len(metadata) == len(sources):
            runtime_groups = _split_runtime_records_by_instance(runtime_log)
            if len(runtime_groups) < len(instances):
                real_metric_note = (
                    f"skipped: runtime source groups {len(runtime_groups)} < instances {len(instances)}"
                )
            else:
                indices_by_paper: Dict[str, List[int]] = {}
                intervals_by_paper: Dict[str, List[Tuple[float, float]]] = {}
                for idx, (paper_id, start, end) in enumerate(metadata):
                    indices_by_paper.setdefault(paper_id, []).append(idx)
                    intervals_by_paper.setdefault(paper_id, []).append((start, end))
                sentence_term_map_keys = [set() for _ in sources]
                multi_instance_sentence_groups = {}
                for inst_idx, inst in enumerate(instances):
                    paper_id = _instance_paper_id(inst)
                    if not paper_id or paper_id not in indices_by_paper:
                        real_metric_note = (
                            f"skipped: cannot map instance {inst_idx} source to audio_yaml paper"
                        )
                        sentence_term_map_keys = None
                        multi_instance_sentence_groups = None
                        break
                    sentence_indices = indices_by_paper[paper_id]
                    local_refs = [refs[i] for i in sentence_indices]
                    hyp_segments = _segment_prediction_by_references(
                        str(inst.get("prediction") or ""),
                        local_refs,
                        latency_unit=latency_unit,
                    )
                    local_keys = _runtime_records_to_sentence_term_map_keys(
                        runtime_groups[inst_idx],
                        intervals_by_paper[paper_id],
                    )
                    if len(local_keys) != len(sentence_indices):
                        real_metric_note = (
                            f"skipped: term_map key rows {len(local_keys)} != sentence rows "
                            f"{len(sentence_indices)} for {paper_id}"
                        )
                        sentence_term_map_keys = None
                        multi_instance_sentence_groups = None
                        break
                    multi_instance_sentence_groups[inst_idx] = []
                    for local_idx, global_idx in enumerate(sentence_indices):
                        sentence_term_map_keys[global_idx] = local_keys[local_idx]
                        multi_instance_sentence_groups[inst_idx].append(
                            (
                                global_idx,
                                sources[global_idx],
                                refs[global_idx],
                                hyp_segments[local_idx],
                            )
                        )
                if sentence_term_map_keys is not None:
                    real_metric_note = f"runtime_log_multi_instance:{runtime_log}"
        else:
            real_metric_note = (
                f"skipped: audio metadata {len(metadata)} != source sentences {len(sources)}"
            )
    elif runtime_log is not None and audio_yaml is not None and not can_align_runtime:
        real_metric_note = (
            f"skipped: cannot safely align {len(instances)} instances to {len(sources)} source sentences"
        )
    elif runtime_log is not None or audio_yaml is not None:
        real_metric_note = "skipped: both runtime_log and audio_yaml are required"

    sentence_rows: List[Dict[str, Any]] = []
    adopted_total = 0
    term_total = 0
    sentence_sum = 0.0
    sentence_count = 0
    real_adopted_total = 0
    real_term_total = 0
    real_sentence_sum = 0.0
    real_sentence_count = 0
    source_no_gold_sentence_count = 0
    source_false_copy_sentence_count = 0
    source_false_copy_term_total = 0
    term_map_negative_sentence_count = 0
    term_map_false_copy_sentence_count = 0
    term_map_false_copy_term_total = 0

    if multi_instance_sentence_groups is not None:
        row_iter = []
        for inst_idx in range(len(instances)):
            row_iter.extend(multi_instance_sentence_groups.get(inst_idx, []))
    elif len(instances) == 1 and len(sources) > 1 and len(refs) > 1:
        hyp_segments = _segment_prediction_by_references(
            str(instances[0].get("prediction") or ""),
            refs,
            latency_unit=latency_unit,
        )
        row_iter = [
            (idx, source_text, ref_text, hyp_text)
            for idx, (source_text, ref_text, hyp_text) in enumerate(zip(sources, refs, hyp_segments))
        ]
    else:
        row_iter = []
        for row_idx, inst in enumerate(instances):
            idx_raw = inst.get("index", row_idx)
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                idx = row_idx
            if idx < 0 or idx >= len(sources) or idx >= len(refs):
                idx = row_idx
            if idx < 0 or idx >= len(sources) or idx >= len(refs):
                continue
            row_iter.append((idx, sources[idx], refs[idx], str(inst.get("prediction") or "")))

    for idx, source_text, ref_text, hyp_text in row_iter:
        relevant: List[GlossaryTerm] = []
        source_ref_negative_terms: List[GlossaryTerm] = []
        false_copies: List[GlossaryTerm] = []
        false_seen = set()
        for item in terms:
            source_has = _source_contains(source_text, item.term)
            ref_has = _text_contains(ref_text, item.translation)
            hyp_has = _text_contains(hyp_text, item.translation)
            if source_has and ref_has:
                relevant.append(item)
                continue
            if not source_has and not ref_has and _can_count_false_copy_translation(item.translation):
                source_ref_negative_terms.append(item)
            if hyp_has and not source_has and not ref_has and _can_count_false_copy_translation(item.translation):
                false_key = item.translation
                if false_key not in false_seen:
                    false_seen.add(false_key)
                    false_copies.append(item)

        if source_ref_negative_terms:
            source_no_gold_sentence_count += 1
            if false_copies:
                source_false_copy_sentence_count += 1
                source_false_copy_term_total += len(false_copies)

        adopted = [item for item in relevant if _text_contains(hyp_text, item.translation)]
        aligned_keys = (
            sentence_term_map_keys[idx]
            if sentence_term_map_keys is not None and 0 <= idx < len(sentence_term_map_keys)
            else set()
        )
        real_relevant = [
            item for item in relevant
            if item.match_key in aligned_keys
            or (_normalise_space(item.translation) and any(k[1] == item.match_key[1] for k in aligned_keys))
        ]
        real_adopted = [
            item for item in real_relevant
            if _text_contains(hyp_text, item.translation)
        ]
        term_map_negative_terms: List[Tuple[str, str]] = []
        term_map_false_copies: List[Tuple[str, str]] = []
        term_map_false_seen: Set[str] = set()
        if sentence_term_map_keys is not None:
            for term, translation in sorted(aligned_keys):
                translation_norm = _normalise_space(translation)
                if not translation_norm or not _can_count_false_copy_translation(translation_norm):
                    continue
                source_has = _source_contains(source_text, term)
                ref_has = _text_contains(ref_text, translation_norm)
                # True false-copy negatives must be unsupported by both sides of
                # the aligned sentence.  This avoids penalizing correct term-map
                # usage when the English source contains the term but the human
                # reference uses an abbreviation, paraphrase, or alternate
                # translation.
                if source_has or ref_has:
                    continue
                term_map_negative_terms.append((term, translation_norm))
                if _text_contains(hyp_text, translation_norm) and translation_norm not in term_map_false_seen:
                    term_map_false_seen.add(translation_norm)
                    term_map_false_copies.append((term, translation_norm))
            if term_map_negative_terms:
                term_map_negative_sentence_count += 1
                if term_map_false_copies:
                    term_map_false_copy_sentence_count += 1
                    term_map_false_copy_term_total += len(term_map_false_copies)
        denom = len(relevant)
        numer = len(adopted)
        real_denom = len(real_relevant)
        real_numer = len(real_adopted)
        real_rate = real_numer / real_denom if real_denom > 0 else None
        if real_denom > 0:
            real_sentence_sum += float(real_rate)
            real_sentence_count += 1
            real_adopted_total += real_numer
            real_term_total += real_denom
        if denom <= 0:
            sentence_rows.append(
                {
                    "index": idx,
                    "source": source_text,
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "term_adoption_rate": None,
                    "adopted_terms": 0,
                    "source_terms": 0,
                    "real_term_adopt_rate": real_rate,
                    "real_adopted_terms": real_numer,
                    "real_term_map_source_terms": real_denom,
                    "terms": [],
                    "term_map_terms": [
                        {"term": term, "translation": translation}
                        for term, translation in sorted(aligned_keys)
                    ],
                    "false_copy_terms": [
                        {
                            "term": item.term,
                            "translation": item.translation,
                        }
                        for item in false_copies
                    ],
                    "source_ref_negative_term_count": len(source_ref_negative_terms),
                    "term_map_negative_terms": [
                        {"term": term, "translation": translation}
                        for term, translation in term_map_negative_terms
                    ],
                    "term_map_false_copy_terms": [
                        {"term": term, "translation": translation}
                        for term, translation in term_map_false_copies
                    ],
                }
            )
            continue

        rate = numer / denom
        sentence_sum += rate
        sentence_count += 1
        adopted_total += numer
        term_total += denom
        sentence_rows.append(
            {
                "index": idx,
                "source": source_text,
                "reference": ref_text,
                "hypothesis": hyp_text,
                "term_adoption_rate": rate,
                "adopted_terms": numer,
                "source_terms": denom,
                "real_term_adopt_rate": real_rate,
                "real_adopted_terms": real_numer,
                "real_term_map_source_terms": real_denom,
                "terms": [
                    {
                        "term": item.term,
                        "translation": item.translation,
                        "adopted": item in adopted,
                        "in_aligned_term_map": item in real_relevant,
                        "real_adopted": item in real_adopted,
                    }
                    for item in relevant
                ],
                "term_map_terms": [
                    {"term": term, "translation": translation}
                    for term, translation in sorted(aligned_keys)
                ],
                "false_copy_terms": [
                    {
                        "term": item.term,
                        "translation": item.translation,
                    }
                    for item in false_copies
                ],
                "source_ref_negative_term_count": len(source_ref_negative_terms),
                "term_map_negative_terms": [
                    {"term": term, "translation": translation}
                    for term, translation in term_map_negative_terms
                ],
                "term_map_false_copy_terms": [
                    {"term": term, "translation": translation}
                    for term, translation in term_map_false_copies
                ],
            }
        )

    macro = sentence_sum / sentence_count if sentence_count > 0 else 0.0
    micro = adopted_total / term_total if term_total > 0 else 0.0
    real_macro = real_sentence_sum / real_sentence_count if real_sentence_count > 0 else 0.0
    real_micro = real_adopted_total / real_term_total if real_term_total > 0 else 0.0
    source_false_copy_rate = (
        source_false_copy_sentence_count / source_no_gold_sentence_count
        if source_no_gold_sentence_count > 0 else 0.0
    )
    term_map_fcr_available = sentence_term_map_keys is not None
    term_map_false_copy_rate = (
        term_map_false_copy_sentence_count / term_map_negative_sentence_count
        if term_map_negative_sentence_count > 0 else 0.0
    )
    if fcr_policy == "source_ref_negative_sentence":
        primary_false_copy_rate = source_false_copy_rate
        primary_false_copy_sentences = source_false_copy_sentence_count
        primary_negative_sentences = source_no_gold_sentence_count
        primary_false_copy_terms = source_false_copy_term_total
        fcr_mode = "source_ref_negative_sentence_fixed_glossary"
    elif fcr_policy == "term_map_source_ref_negative_sentence":
        if not term_map_fcr_available:
            details = real_metric_note or (
                "runtime_log and audio_yaml alignment is required for term-map-gated FCR"
            )
            raise ValueError(
                "term_map_source_ref_negative_sentence requires aligned runtime term_map keys; "
                f"{details}"
            )
        primary_false_copy_rate = term_map_false_copy_rate
        primary_false_copy_sentences = term_map_false_copy_sentence_count
        primary_negative_sentences = term_map_negative_sentence_count
        primary_false_copy_terms = term_map_false_copy_term_total
        fcr_mode = "term_map_source_ref_negative_sentence"
    else:
        primary_false_copy_rate = term_map_false_copy_rate if term_map_fcr_available else source_false_copy_rate
        primary_false_copy_sentences = (
            term_map_false_copy_sentence_count
            if term_map_fcr_available else source_false_copy_sentence_count
        )
        primary_negative_sentences = (
            term_map_negative_sentence_count
            if term_map_fcr_available else source_no_gold_sentence_count
        )
        primary_false_copy_terms = (
            term_map_false_copy_term_total
            if term_map_fcr_available else source_false_copy_term_total
        )
        fcr_mode = (
            "term_map_sentence_true_false_copy"
            if term_map_fcr_available else "source_ref_negative_sentence_fixed_glossary_fallback"
        )
    return {
        "term_adoption": macro,
        "term_adoption_micro": micro,
        "adopted": adopted_total,
        "total": term_total,
        "sentence_count": sentence_count,
        "real_term_adopt": real_macro,
        "real_term_adopt_micro": real_micro,
        "real_adopted": real_adopted_total,
        "real_total": real_term_total,
        "real_sentence_count": real_sentence_count,
        "real_metric_available": sentence_term_map_keys is not None,
        "real_metric_note": real_metric_note,
        "term_fcr": primary_false_copy_rate,
        "false_copy_sentences": primary_false_copy_sentences,
        "no_gold_sentences": primary_negative_sentences,
        "false_copy_terms": primary_false_copy_terms,
        "term_fcr_mode": fcr_mode,
        "term_map_fcr": term_map_false_copy_rate,
        "term_map_false_copy_sentences": term_map_false_copy_sentence_count,
        "term_map_negative_sentences": term_map_negative_sentence_count,
        "term_map_false_copy_terms": term_map_false_copy_term_total,
        "source_term_sent_fcr": source_false_copy_rate,
        "source_false_copy_sentences": source_false_copy_sentence_count,
        "source_no_gold_sentences": source_no_gold_sentence_count,
        "source_false_copy_terms": source_false_copy_term_total,
        "instance_count": len(instances),
        "glossary_terms": len(terms),
        "sentences": sentence_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances-log", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--glossary-path", required=True)
    parser.add_argument("--target-lang", default=TARGET_LANG_DEFAULT)
    parser.add_argument("--latency-unit", default="char", choices=["word", "char"])
    parser.add_argument("--runtime-log", default="")
    parser.add_argument("--audio-yaml", default="")
    parser.add_argument(
        "--sentence-term-map",
        default="",
        help=(
            "Optional sentence-indexed term_map JSON. When provided, REAL_TERM_ADOPT "
            "and term-map-gated FCR use this explicit sentence map instead of "
            "timeline-aligning runtime prompt records."
        ),
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--fcr-policy",
        choices=[
            "term_map_if_available",
            "term_map_source_ref_negative_sentence",
            "source_ref_negative_sentence",
        ],
        default="term_map_if_available",
        help=(
            "TERM_SENT_FCR policy. Use term_map_source_ref_negative_sentence for "
            "oracle/readout metrics where false-copy candidates must be exposed "
            "by the aligned runtime term_map."
        ),
    )
    args = parser.parse_args()

    result = compute_sentence_adoption(
        instances_log=Path(args.instances_log),
        source_file=Path(args.source_file),
        reference_file=Path(args.reference_file),
        glossary_path=Path(args.glossary_path),
        target_lang=args.target_lang,
        latency_unit=args.latency_unit,
        runtime_log=Path(args.runtime_log) if args.runtime_log else None,
        audio_yaml=Path(args.audio_yaml) if args.audio_yaml else None,
        sentence_term_map=Path(args.sentence_term_map) if args.sentence_term_map else None,
        fcr_policy=args.fcr_policy,
    )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "TERM_ADOPTION\t{macro:.6f}\tADOPTED\t{adopted:d}\tTOTAL\t{total:d}"
        "\tSENTENCES\t{sentences:d}\tMICRO\t{micro:.6f}".format(
            macro=float(result["term_adoption"]),
            adopted=int(result["adopted"]),
            total=int(result["total"]),
            sentences=int(result["sentence_count"]),
            micro=float(result["term_adoption_micro"]),
        )
    )
    if bool(result.get("real_metric_available", False)):
        print(
            "REAL_TERM_ADOPT\t{macro:.6f}\tADOPTED\t{adopted:d}\tTOTAL\t{total:d}"
            "\tSENTENCES\t{sentences:d}\tMICRO\t{micro:.6f}".format(
                macro=float(result["real_term_adopt"]),
                adopted=int(result["real_adopted"]),
                total=int(result["real_total"]),
                sentences=int(result["real_sentence_count"]),
                micro=float(result["real_term_adopt_micro"]),
            )
        )
    else:
        note = _normalise_space(result.get("real_metric_note") or "runtime_log/audio_yaml unavailable")
        print(f"REAL_TERM_ADOPT\tN/A\tNOTE\t{note}")
    print(
        "TERM_SENT_FCR\t{fcr:.6f}\tFALSE_COPY_SENTENCES\t{false_sentences:d}"
        "\tNEGATIVE_SENTENCES\t{no_gold:d}\tFALSE_COPY_TERMS\t{false_terms:d}"
        "\tMODE\t{mode}".format(
            fcr=float(result["term_fcr"]),
            false_sentences=int(result["false_copy_sentences"]),
            no_gold=int(result["no_gold_sentences"]),
            false_terms=int(result["false_copy_terms"]),
            mode=str(result["term_fcr_mode"]),
        )
    )
    print(
        "SOURCE_TERM_SENT_FCR\t{fcr:.6f}\tFALSE_COPY_SENTENCES\t{false_sentences:d}"
        "\tNO_GT_SENTENCES\t{no_gold:d}\tFALSE_COPY_TERMS\t{false_terms:d}".format(
            fcr=float(result["source_term_sent_fcr"]),
            false_sentences=int(result["source_false_copy_sentences"]),
            no_gold=int(result["source_no_gold_sentences"]),
            false_terms=int(result["source_false_copy_terms"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
