#!/usr/bin/env python3
"""Register and query experiment events in the local SQLite index.

This is the provenance layer around WandB runs.  WandB remains authoritative for
metrics/config/verdict; event manifests record launchers, commands, data paths,
logs, and downstream links that WandB does not reliably reconstruct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    from experiment_db import default_db_path, json_dumps, utc_now
except ImportError:  # pragma: no cover - only for unusual direct imports.
    sys.path.append(str(Path(__file__).resolve().parent))
    from experiment_db import default_db_path, json_dumps, utc_now

try:
    from wandb_tags import MAX_WANDB_TAG_LEN
except ImportError:  # pragma: no cover - only for unusual direct imports.
    sys.path.append(str(Path(__file__).resolve().parent))
    from wandb_tags import MAX_WANDB_TAG_LEN


REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = Path(__file__).with_name("experiment_db_schema.sql")
JOB_ID_RE = re.compile(r"\b(\d{3,})\b")
WAND_B_TAG_ENV_KEYS = {
    "EXPERIMENT_FAMILY": "family",
    "TASK_TAG": "task",
    "DATA_TAG": "data",
    "VARIANT_TAG": "variant",
}
EXPORT_RE = re.compile(
    r"^\s*(?:export\s+)?"
    r"(EXPERIMENT_FAMILY|TASK_TAG|DATA_TAG|VARIANT_TAG|EXTRA_WANDB_TAGS)"
    r"=(.*)$"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_local_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def path_exists(path_text: str) -> Optional[int]:
    if not path_text:
        return None
    if "://" in path_text:
        return None
    return 1 if resolve_local_path(path_text).exists() else 0


def open_db(db_path: Optional[str]) -> sqlite3.Connection:
    path = Path(db_path).expanduser() if db_path else default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(schema)
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (1, utc_now()),
    )
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (2, utc_now()),
    )
    conn.commit()
    return conn


def git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return out.strip() or None


def git_dirty() -> int:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return 0
    return 1 if out.strip() else 0


def load_manifest(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    for key in ("event_id", "event_type"):
        if not data.get(key):
            raise ValueError(f"manifest missing required key `{key}`: {path}")
    return data


def _strip_shell_comment(text: str) -> str:
    """Remove simple shell comments outside single/double quotes."""
    out: List[str] = []
    quote: Optional[str] = None
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if quote:
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _unquote_shell_value(raw: str) -> str:
    text = _strip_shell_comment(raw).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    # Handle the common launcher idiom: "${DATA_TAG:-short_default}".
    default_match = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:-(.*)\}", text)
    if default_match:
        text = default_match.group(1)
    return text.strip()


def _split_tag_words(text: str) -> List[str]:
    try:
        return [part for part in shlex.split(text) if part]
    except ValueError:
        return [part for part in text.split() if part]


def _launcher_tag_candidates(launcher_path: str) -> List[str]:
    if not launcher_path:
        return []
    path = resolve_local_path(launcher_path)
    if not path.exists() or not path.is_file():
        return []
    tags: List[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EXPORT_RE.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = _unquote_shell_value(raw_value)
        if not value or "$" in value:
            # Do not guess dynamic shell expansions. Static defaults are handled above.
            continue
        if key == "EXTRA_WANDB_TAGS":
            tags.extend(_split_tag_words(value))
            continue
        prefix = WAND_B_TAG_ENV_KEYS[key]
        tags.append(f"{prefix}:{value}")
    return tags


def _metadata_tag_candidates(metadata: Mapping[str, Any]) -> List[str]:
    tags: List[str] = []
    key_map = {
        "experiment_family": "family",
        "family_tag": "family",
        "task_tag": "task",
        "data_tag": "data",
        "variant_tag": "variant",
    }
    for key, prefix in key_map.items():
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            tags.append(f"{prefix}:{value.strip()}")
    for key in ("wandb_tags", "extra_wandb_tags"):
        value = metadata.get(key)
        if isinstance(value, str):
            tags.extend(_split_tag_words(value))
        elif isinstance(value, Iterable):
            tags.extend(str(item).strip() for item in value if str(item).strip())
    return tags


def validate_manifest_wandb_tags(
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Fail before launch/register if static manifest or launcher tags are invalid."""
    candidates: List[str] = []
    family = str(manifest.get("family") or "").strip()
    if family:
        candidates.append(f"family:{family}")
    metadata = manifest.get("metadata") or {}
    if isinstance(metadata, Mapping):
        candidates.extend(_metadata_tag_candidates(metadata))
    candidates.extend(_launcher_tag_candidates(str(manifest.get("launcher_path") or "")))

    seen = set()
    unique = []
    for tag in candidates:
        tag = str(tag).strip()
        if tag and tag not in seen:
            unique.append(tag)
            seen.add(tag)

    bad = [tag for tag in unique if not (1 <= len(tag) <= MAX_WANDB_TAG_LEN)]
    if not bad:
        return
    details = "\n".join(f"  - len={len(tag)} {tag}" for tag in bad)
    raise ValueError(
        f"manifest has WandB tag(s) outside 1..{MAX_WANDB_TAG_LEN} chars: "
        f"{manifest_path}\n{details}\n"
        "Shorten DATA_TAG/VARIANT_TAG/EXTRA_WANDB_TAGS in the launcher or "
        "metadata before registering/launching. Use long names in "
        "WANDB_EXP_NAME, notes, manifest metadata, or artifact paths instead."
    )


def normalize_run_ids(raw: Any) -> List[str]:
    ids: List[str] = []
    if raw is None:
        return ids
    if isinstance(raw, str):
        ids.extend(x for x in re.split(r"[\s,]+", raw) if x)
    elif isinstance(raw, Iterable):
        for item in raw:
            if item is not None:
                ids.append(str(item))
    out: List[str] = []
    seen = set()
    for rid in ids:
        rid = rid.strip()
        if rid and rid not in seen:
            out.append(rid)
            seen.add(rid)
    return out


def artifact_rows(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in manifest.get("artifacts", []) or []:
        if not isinstance(item, Mapping):
            continue
        path = str(item.get("path") or item.get("path_or_uri") or "").strip()
        if not path:
            continue
        rows.append({
            "role": str(item.get("role") or item.get("artifact_type") or "artifact"),
            "artifact_type": str(item.get("type") or item.get("artifact_type") or "file"),
            "direction": str(item.get("direction") or "unknown"),
            "path_or_uri": path,
            "metadata": dict(item.get("metadata") or {}),
        })

    for role, artifact_type, key in (
        ("launcher", "script", "launcher_path"),
        ("notes", "notes", "notes_path"),
    ):
        path = str(manifest.get(key) or "").strip()
        if path and not any(r["role"] == role and r["path_or_uri"] == path for r in rows):
            rows.append({
                "role": role,
                "artifact_type": artifact_type,
                "direction": "control",
                "path_or_uri": path,
                "metadata": {"manifest_key": key},
            })
    return rows


def upsert_manifest(
    conn: sqlite3.Connection,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    command_override: Optional[str] = None,
    status_override: Optional[str] = None,
    slurm_job_id_override: Optional[str] = None,
) -> None:
    now = utc_now()
    raw = manifest_path.read_text(encoding="utf-8")
    event_id = str(manifest["event_id"])
    launcher_path = str(manifest.get("launcher_path") or "")
    launcher_sha = str(manifest.get("launcher_sha256") or "")
    if launcher_path and not launcher_sha:
        local_launcher = resolve_local_path(launcher_path)
        if local_launcher.exists() and local_launcher.is_file():
            launcher_sha = sha256_file(local_launcher)

    metadata = dict(manifest.get("metadata") or {})
    metadata.setdefault("schema_version", manifest.get("schema_version", 1))
    parent_run_ids = normalize_run_ids(
        manifest.get("parent_run_ids") or manifest.get("baseline_run_ids")
    )
    if parent_run_ids:
        metadata.setdefault("parent_run_ids", parent_run_ids)

    status = status_override or str(manifest.get("status") or "planned")
    command = command_override if command_override is not None else manifest.get("command")
    slurm_job_id = slurm_job_id_override if slurm_job_id_override is not None else manifest.get("slurm_job_id")
    wandb_run_id = manifest.get("wandb_run_id") or None
    dirty_value = manifest.get("git_dirty")
    if dirty_value is None:
        dirty_value = git_dirty()

    with conn:
        conn.execute(
            """
            INSERT INTO experiment_events(
                event_id, event_type, family, variant, status, project,
                wandb_run_id, slurm_job_id, launcher_path, launcher_sha256,
                notes_path, git_commit, git_dirty, cwd, command, manifest_path,
                manifest_sha256, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                event_type=excluded.event_type,
                family=excluded.family,
                variant=excluded.variant,
                status=excluded.status,
                project=excluded.project,
                wandb_run_id=excluded.wandb_run_id,
                slurm_job_id=excluded.slurm_job_id,
                launcher_path=excluded.launcher_path,
                launcher_sha256=excluded.launcher_sha256,
                notes_path=excluded.notes_path,
                git_commit=excluded.git_commit,
                git_dirty=excluded.git_dirty,
                cwd=excluded.cwd,
                command=excluded.command,
                manifest_path=excluded.manifest_path,
                manifest_sha256=excluded.manifest_sha256,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (
                event_id,
                str(manifest["event_type"]),
                manifest.get("family"),
                manifest.get("variant"),
                status,
                manifest.get("project"),
                wandb_run_id,
                slurm_job_id,
                launcher_path or None,
                launcher_sha or None,
                manifest.get("notes_path") or None,
                manifest.get("git_commit") or git_commit(),
                int(dirty_value),
                manifest.get("cwd") or str(REPO_ROOT),
                command,
                str(manifest_path),
                sha256_text(raw),
                json_dumps(metadata),
                manifest.get("created_at") or now,
                now,
            ),
        )

        conn.execute("DELETE FROM experiment_event_artifacts WHERE event_id = ?", (event_id,))
        conn.executemany(
            """
            INSERT INTO experiment_event_artifacts(
                event_id, role, artifact_type, direction, path_or_uri,
                exists_local, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event_id,
                    row["role"],
                    row["artifact_type"],
                    row["direction"],
                    row["path_or_uri"],
                    path_exists(row["path_or_uri"]),
                    json_dumps(row["metadata"]),
                )
                for row in artifact_rows(manifest)
            ],
        )

        conn.execute("DELETE FROM experiment_run_links WHERE event_id = ?", (event_id,))
        if wandb_run_id:
            conn.executemany(
                """
                INSERT OR REPLACE INTO experiment_run_links(
                    parent_run_id, child_run_id, link_type, event_id,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        parent_id,
                        wandb_run_id,
                        "parent",
                        event_id,
                        json_dumps({"source": "event_manifest"}),
                        now,
                    )
                    for parent_id in parent_run_ids
                ],
            )


def row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def cmd_init(args: argparse.Namespace) -> int:
    conn = open_db(args.db_path)
    conn.close()
    print(f"[experiment_event] initialized {args.db_path or default_db_path()}")
    return 0


def cmd_register(args: argparse.Namespace) -> int:
    conn = open_db(args.db_path)
    for manifest_arg in args.manifests:
        manifest_path = resolve_local_path(manifest_arg)
        manifest = load_manifest(manifest_path)
        validate_manifest_wandb_tags(manifest, manifest_path)
        upsert_manifest(conn, manifest_path, manifest)
        print(f"[experiment_event] registered {manifest['event_id']}")
    conn.close()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    conn = open_db(args.db_path)
    row = conn.execute(
        """
        SELECT * FROM experiment_events
        WHERE event_id = ? OR wandb_run_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (args.id, args.id),
    ).fetchone()
    if row is None:
        print(f"[experiment_event] no event found for {args.id}", file=sys.stderr)
        return 1
    event = row_to_dict(row)
    event["metadata"] = json.loads(event.pop("metadata_json") or "{}")
    artifacts = conn.execute(
        """
        SELECT role, artifact_type, direction, path_or_uri, exists_local, metadata_json
        FROM experiment_event_artifacts
        WHERE event_id = ?
        ORDER BY direction, role, path_or_uri
        """,
        (event["event_id"],),
    ).fetchall()
    event["artifacts"] = [
        {
            **{k: v for k, v in row_to_dict(a).items() if k != "metadata_json"},
            "metadata": json.loads(a["metadata_json"] or "{}"),
        }
        for a in artifacts
    ]
    print(json.dumps(event, ensure_ascii=False, indent=2))
    return 0


def cmd_files(args: argparse.Namespace) -> int:
    conn = open_db(args.db_path)
    event = conn.execute(
        """
        SELECT event_id FROM experiment_events
        WHERE event_id = ? OR wandb_run_id = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (args.id, args.id),
    ).fetchone()
    if event is None:
        print(f"[experiment_event] no event found for {args.id}", file=sys.stderr)
        return 1
    roles = set(args.role or [])
    rows = conn.execute(
        """
        SELECT role, artifact_type, direction, exists_local, path_or_uri
        FROM experiment_event_artifacts
        WHERE event_id = ?
        ORDER BY direction, role, path_or_uri
        """,
        (event["event_id"],),
    ).fetchall()
    for row in rows:
        if roles and row["role"] not in roles:
            continue
        exists = "?" if row["exists_local"] is None else ("yes" if row["exists_local"] else "no")
        print(
            f"{row['direction']}\t{row['role']}\t{row['artifact_type']}"
            f"\texists={exists}\t{row['path_or_uri']}"
        )
    return 0


def cmd_find(args: argparse.Namespace) -> int:
    conn = open_db(args.db_path)
    where: List[str] = []
    params: List[Any] = []
    if args.event_type:
        where.append("e.event_type = ?")
        params.append(args.event_type)
    if args.family:
        where.append("e.family = ?")
        params.append(args.family)
    if args.status:
        where.append("e.status = ?")
        params.append(args.status)
    if args.variant_contains:
        where.append("e.variant LIKE ?")
        params.append(f"%{args.variant_contains}%")
    if args.path_contains:
        where.append(
            """
            EXISTS (
                SELECT 1 FROM experiment_event_artifacts a
                WHERE a.event_id = e.event_id AND a.path_or_uri LIKE ?
            )
            """
        )
        params.append(f"%{args.path_contains}%")
    sql = """
        SELECT e.event_id, e.event_type, e.family, e.variant, e.status,
               e.project, e.wandb_run_id, e.slurm_job_id, e.launcher_path,
               e.updated_at
        FROM experiment_events e
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY e.updated_at DESC LIMIT ?"
    params.append(args.limit)
    rows = conn.execute(sql, params).fetchall()
    for row in rows:
        print(
            f"{row['updated_at']}\t{row['event_id']}\t{row['event_type']}"
            f"\t{row['status']}\t{row['wandb_run_id'] or '-'}\t{row['launcher_path'] or '-'}"
        )
    return 0


def parse_job_id(stdout: str) -> Optional[str]:
    for line in stdout.splitlines():
        m = JOB_ID_RE.search(line)
        if m:
            return m.group(1)
    return None


def cmd_launch(args: argparse.Namespace) -> int:
    if not args.command:
        print("[experiment_event] launch requires a command after --", file=sys.stderr)
        return 2
    manifest_path = resolve_local_path(args.manifest)
    manifest = load_manifest(manifest_path)
    validate_manifest_wandb_tags(manifest, manifest_path)
    command_text = " ".join(args.command)
    conn = open_db(args.db_path)
    upsert_manifest(
        conn,
        manifest_path,
        manifest,
        command_override=command_text,
        status_override="launching",
    )
    print(f"[experiment_event] launching {manifest['event_id']}: {command_text}")
    proc = subprocess.run(
        args.command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    job_id = parse_job_id(proc.stdout)
    upsert_manifest(
        conn,
        manifest_path,
        manifest,
        command_override=command_text,
        status_override="submitted" if proc.returncode == 0 else "launch_failed",
        slurm_job_id_override=job_id,
    )
    conn.close()
    if proc.returncode == 0:
        print(f"[experiment_event] submitted event={manifest['event_id']} slurm_job_id={job_id or '-'}")
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-path", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="Create or upgrade event tables.")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("register", help="Register one or more JSON manifests.")
    sp.add_argument("manifests", nargs="+")
    sp.set_defaults(func=cmd_register)

    sp = sub.add_parser("show", help="Show one event by event id or WandB run id.")
    sp.add_argument("id")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("files", help="List files/artifacts linked to an event or run.")
    sp.add_argument("id")
    sp.add_argument("--role", action="append", default=None)
    sp.set_defaults(func=cmd_files)

    sp = sub.add_parser("find", help="Find registered events.")
    sp.add_argument("--event-type", default=None)
    sp.add_argument("--family", default=None)
    sp.add_argument("--status", default=None)
    sp.add_argument("--variant-contains", default=None)
    sp.add_argument("--path-contains", default=None)
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(func=cmd_find)

    sp = sub.add_parser("launch", help="Register a manifest, run a command, and store the Slurm id if present.")
    sp.add_argument("manifest")
    sp.add_argument("command", nargs=argparse.REMAINDER)
    sp.set_defaults(func=cmd_launch)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command[0:1] == ["--"]:
        args.command = args.command[1:]
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
