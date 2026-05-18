#!/usr/bin/env python3
"""SQLite experiment index derived from WandB and run notes.

This module intentionally does not define experiment truth.  It stores a
rebuildable local index that helps humans and agents find runs, notes, configs,
baselines, and at-best-step metric bundles without depending on chat memory.
Metrics are imported only from WandB reads performed by `wandb_tool.py`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = REPO_ROOT / "documents" / "code" / ".cache" / "experiments.sqlite"
SCHEMA_PATH = Path(__file__).with_name("experiment_db_schema.sql")
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
WANDB_RUN_RE = re.compile(r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/([A-Za-z0-9_-]+)")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_db_path() -> Path:
    return Path(os.environ.get("EXPERIMENT_DB_PATH", DEFAULT_DB_PATH)).expanduser()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def json_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json_dumps(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_tag(tags: Sequence[str], prefix: str) -> Optional[str]:
    for tag in tags:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return None


def parse_notes_sections(notes: str) -> Dict[str, str]:
    """Parse top-level `## Section` blocks from run notes."""
    matches = list(SECTION_RE.finditer(notes or ""))
    sections: Dict[str, str] = {}
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(notes)
        sections[title] = notes[start:end].strip()
    return sections


def infer_notes_path(config: Mapping[str, Any], notes: str) -> Optional[str]:
    path = config.get("notes_file")
    if isinstance(path, str) and path:
        return path
    for line in (notes or "").splitlines():
        if "notes_" in line and ".md" in line:
            m = re.search(r"([A-Za-z0-9_./-]*notes_[A-Za-z0-9_./-]+\.md)", line)
            if m:
                return m.group(1)
    return None


def baseline_ids_from(config: Mapping[str, Any], notes: str) -> List[str]:
    raw = config.get("baseline_run_ids", [])
    ids: List[str] = []
    if isinstance(raw, str):
        ids.extend(x for x in re.split(r"[\s,]+", raw) if x)
    elif isinstance(raw, Iterable):
        for item in raw:
            if item is not None:
                ids.append(str(item))
    for rid in WANDB_RUN_RE.findall(notes or ""):
        if rid not in ids:
            ids.append(rid)
    # Preserve order while dropping empties/duplicates.
    out: List[str] = []
    seen = set()
    for rid in ids:
        rid = rid.strip()
        if rid and rid not in seen:
            out.append(rid)
            seen.add(rid)
    return out


def artifact_rows_from(config: Mapping[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    for key in (
        "save_path",
        "resume",
        "rag_model_path",
        "output_base",
        "train_jsonl",
        "dev_jsonl",
        "acl_dev_jsonl",
        "eval_wiki_glossary",
    ):
        value = config.get(key)
        if isinstance(value, str) and value:
            rows.append((key, value, {"config_key": key}))
    return rows


class ExperimentDB:
    def __init__(self, path: Optional[os.PathLike[str] | str] = None) -> None:
        self.path = Path(path).expanduser() if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        self.conn.executescript(schema)
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (1, utc_now()),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (2, utc_now()),
        )
        self.conn.commit()

    def log_event(
        self,
        *,
        run_id: Optional[str],
        project: Optional[str],
        source: str,
        command: Optional[str],
        status: str,
        message: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO sync_events(run_id, project, source, command, status, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, project, source, command, status, message, utc_now()),
        )
        self.conn.commit()

    def upsert_run(
        self,
        *,
        run_id: str,
        entity: str,
        project: str,
        name: str,
        url: str,
        state: Optional[str],
        created_at: Optional[str],
        runtime_s: Optional[float],
        tags: Sequence[str],
        config: Mapping[str, Any],
        notes: str,
        summary: Mapping[str, Any],
    ) -> None:
        self.init_schema()
        now = utc_now()
        family = extract_tag(tags, "family:")
        task = extract_tag(tags, "task:")
        data_tag = extract_tag(tags, "data:")
        variant_tag = extract_tag(tags, "variant:")
        status_tag = extract_tag(tags, "status:")
        notes_path = infer_notes_path(config, notes)
        source_hash = sha256_text(json_dumps({
            "run_id": run_id,
            "entity": entity,
            "project": project,
            "name": name,
            "state": state,
            "tags": list(tags),
            "config": dict(config),
            "notes_sha256": sha256_text(notes or ""),
            "verdict": summary.get("verdict"),
        }))

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs(
                    run_id, entity, project, name, url, state, created_at, runtime_s,
                    tags_json, family, task, data_tag, variant_tag, status_tag,
                    notes_path, notes_sha256, summary_verdict, source_hash, synced_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    entity=excluded.entity,
                    project=excluded.project,
                    name=excluded.name,
                    url=excluded.url,
                    state=excluded.state,
                    created_at=excluded.created_at,
                    runtime_s=excluded.runtime_s,
                    tags_json=excluded.tags_json,
                    family=excluded.family,
                    task=excluded.task,
                    data_tag=excluded.data_tag,
                    variant_tag=excluded.variant_tag,
                    status_tag=excluded.status_tag,
                    notes_path=excluded.notes_path,
                    notes_sha256=excluded.notes_sha256,
                    summary_verdict=excluded.summary_verdict,
                    source_hash=excluded.source_hash,
                    synced_at=excluded.synced_at
                """,
                (
                    run_id, entity, project, name, url, state, created_at, runtime_s,
                    json_dumps(list(tags)), family, task, data_tag, variant_tag, status_tag,
                    notes_path, sha256_text(notes or ""), summary.get("verdict"), source_hash, now,
                ),
            )

            self.conn.execute("DELETE FROM run_config WHERE run_id = ?", (run_id,))
            self.conn.executemany(
                """
                INSERT INTO run_config(run_id, key, value_json, value_text)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (run_id, key, json_dumps(value), json_text(value))
                    for key, value in sorted(config.items())
                    if not str(key).startswith("_")
                ],
            )

            self.conn.execute("DELETE FROM notes_sections WHERE run_id = ?", (run_id,))
            self.conn.executemany(
                """
                INSERT INTO notes_sections(run_id, section, content)
                VALUES (?, ?, ?)
                """,
                [
                    (run_id, section, content)
                    for section, content in parse_notes_sections(notes or "").items()
                ],
            )

            self.conn.execute("DELETE FROM baselines WHERE run_id = ?", (run_id,))
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO baselines(run_id, baseline_run_id, ordinal)
                VALUES (?, ?, ?)
                """,
                [
                    (run_id, baseline_id, idx)
                    for idx, baseline_id in enumerate(baseline_ids_from(config, notes))
                ],
            )

            self.conn.execute("DELETE FROM artifacts WHERE run_id = ?", (run_id,))
            self.conn.executemany(
                """
                INSERT OR REPLACE INTO artifacts(run_id, artifact_type, path_or_uri, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (run_id, artifact_type, path, json_dumps(metadata))
                    for artifact_type, path, metadata in artifact_rows_from(config)
                ],
            )

    def upsert_metric_bundle(
        self,
        run_id: str,
        anchor: str,
        anchor_step: Optional[int],
        metrics: Mapping[str, Optional[float]],
        *,
        source: str = "wandb_history_at_best_step",
    ) -> None:
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "DELETE FROM metrics_at_best WHERE run_id = ? AND anchor = ?",
                (run_id, anchor),
            )
            self.conn.executemany(
                """
                INSERT INTO metrics_at_best(
                    run_id, anchor, anchor_step, metric_key, metric_value, source, synced_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (run_id, anchor, anchor_step, key, value, source, now)
                    for key, value in metrics.items()
                    if value is not None
                ],
            )

    def find_runs(
        self,
        *,
        run_ids: Sequence[str] = (),
        family: Optional[str] = None,
        project: Optional[str] = None,
        status: Optional[str] = None,
        data_tag: Optional[str] = None,
        variant_contains: Optional[str] = None,
        name_contains: Optional[str] = None,
        config_filters: Sequence[str] = (),
        limit: int = 50,
    ) -> List[sqlite3.Row]:
        self.init_schema()
        where: List[str] = []
        params: List[Any] = []
        if run_ids:
            placeholders = ",".join("?" for _ in run_ids)
            where.append(f"r.run_id IN ({placeholders})")
            params.extend(run_ids)
        if family:
            where.append("r.family = ?")
            params.append(family)
        if project:
            where.append("r.project = ?")
            params.append(project)
        if status:
            where.append("r.status_tag = ?")
            params.append(status.removeprefix("status:"))
        if data_tag:
            where.append("r.data_tag = ?")
            params.append(data_tag)
        if variant_contains:
            where.append("COALESCE(r.variant_tag, '') LIKE ?")
            params.append(f"%{variant_contains}%")
        if name_contains:
            where.append("r.name LIKE ?")
            params.append(f"%{name_contains}%")
        joins = ""
        for idx, filt in enumerate(config_filters):
            if "=" not in filt:
                raise ValueError(f"config filter must be key=value: {filt}")
            key, value = filt.split("=", 1)
            alias = f"c{idx}"
            joins += f" JOIN run_config {alias} ON {alias}.run_id = r.run_id"
            where.append(f"{alias}.key = ? AND {alias}.value_text = ?")
            params.extend([key, value])
        sql = "SELECT r.* FROM runs r" + joins
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY COALESCE(r.created_at, '') DESC, r.synced_at DESC LIMIT ?"
        params.append(limit)
        return list(self.conn.execute(sql, params))

    def get_run(self, run_id: str) -> Optional[sqlite3.Row]:
        self.init_schema()
        return self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def config_for(self, run_id: str) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT key, value_json FROM run_config WHERE run_id = ? ORDER BY key",
            (run_id,),
        ).fetchall()
        out: Dict[str, Any] = {}
        for row in rows:
            try:
                out[row["key"]] = json.loads(row["value_json"])
            except Exception:
                out[row["key"]] = row["value_json"]
        return out

    def notes_for(self, run_id: str) -> Dict[str, str]:
        rows = self.conn.execute(
            "SELECT section, content FROM notes_sections WHERE run_id = ? ORDER BY section",
            (run_id,),
        ).fetchall()
        return {row["section"]: row["content"] for row in rows}

    def baselines_for(self, run_id: str) -> List[str]:
        rows = self.conn.execute(
            "SELECT baseline_run_id FROM baselines WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        return [row["baseline_run_id"] for row in rows]

    def metrics_for(self, run_id: str, anchor: Optional[str] = None) -> List[sqlite3.Row]:
        if anchor:
            return list(self.conn.execute(
                """
                SELECT * FROM metrics_at_best
                WHERE run_id = ? AND anchor = ?
                ORDER BY metric_key
                """,
                (run_id, anchor),
            ))
        return list(self.conn.execute(
            "SELECT * FROM metrics_at_best WHERE run_id = ? ORDER BY anchor, metric_key",
            (run_id,),
        ))

    def doctor(self) -> Dict[str, List[str]]:
        self.init_schema()
        issues: Dict[str, List[str]] = {
            "missing_status": [],
            "missing_family": [],
            "missing_notes": [],
            "empty_verdict": [],
            "overlong_tags": [],
            "missing_best_metrics": [],
        }
        rows = self.conn.execute("SELECT * FROM runs ORDER BY project, run_id").fetchall()
        for row in rows:
            rid = row["run_id"]
            tags = json.loads(row["tags_json"] or "[]")
            if not row["status_tag"]:
                issues["missing_status"].append(rid)
            if not row["family"]:
                issues["missing_family"].append(rid)
            if not row["notes_sha256"] or row["notes_sha256"] == sha256_text(""):
                issues["missing_notes"].append(rid)
            if not (row["summary_verdict"] or "").strip():
                verdict = self.notes_for(rid).get("Verdict", "")
                if not verdict.strip() or verdict.strip().upper().startswith("PENDING"):
                    issues["empty_verdict"].append(rid)
            bad_tags = [tag for tag in tags if not (1 <= len(tag) <= 64)]
            if bad_tags:
                issues["overlong_tags"].append(f"{rid}: {bad_tags}")
            n_metrics = self.conn.execute(
                "SELECT COUNT(*) AS n FROM metrics_at_best WHERE run_id = ?",
                (rid,),
            ).fetchone()["n"]
            if n_metrics == 0:
                issues["missing_best_metrics"].append(rid)
        return issues
