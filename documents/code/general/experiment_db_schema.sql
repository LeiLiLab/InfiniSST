PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    entity TEXT NOT NULL,
    project TEXT NOT NULL,
    name TEXT NOT NULL,
    url TEXT NOT NULL,
    state TEXT,
    created_at TEXT,
    runtime_s REAL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    family TEXT,
    task TEXT,
    data_tag TEXT,
    variant_tag TEXT,
    status_tag TEXT,
    notes_path TEXT,
    notes_sha256 TEXT,
    summary_verdict TEXT,
    source_hash TEXT,
    synced_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_project_family ON runs(project, family);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status_tag);
CREATE INDEX IF NOT EXISTS idx_runs_data_variant ON runs(data_tag, variant_tag);

CREATE TABLE IF NOT EXISTS run_config (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value_json TEXT NOT NULL,
    value_text TEXT,
    PRIMARY KEY (run_id, key)
);

CREATE INDEX IF NOT EXISTS idx_run_config_key_text ON run_config(key, value_text);

CREATE TABLE IF NOT EXISTS notes_sections (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    content TEXT NOT NULL,
    PRIMARY KEY (run_id, section)
);

CREATE TABLE IF NOT EXISTS baselines (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    baseline_run_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, baseline_run_id)
);

CREATE INDEX IF NOT EXISTS idx_baselines_baseline ON baselines(baseline_run_id);

CREATE TABLE IF NOT EXISTS metrics_at_best (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    anchor TEXT NOT NULL,
    anchor_step INTEGER,
    metric_key TEXT NOT NULL,
    metric_value REAL,
    source TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (run_id, anchor, metric_key)
);

CREATE INDEX IF NOT EXISTS idx_metrics_at_best_key ON metrics_at_best(metric_key);

CREATE TABLE IF NOT EXISTS metrics_history_index (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step INTEGER NOT NULL,
    metric_key TEXT NOT NULL,
    metric_value REAL,
    source TEXT NOT NULL,
    synced_at TEXT NOT NULL,
    PRIMARY KEY (run_id, step, metric_key)
);

CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    path_or_uri TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (run_id, artifact_type, path_or_uri)
);

CREATE TABLE IF NOT EXISTS sync_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    project TEXT,
    source TEXT NOT NULL,
    command TEXT,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    family TEXT,
    variant TEXT,
    status TEXT NOT NULL DEFAULT 'planned',
    project TEXT,
    wandb_run_id TEXT REFERENCES runs(run_id) ON DELETE SET NULL,
    slurm_job_id TEXT,
    launcher_path TEXT,
    launcher_sha256 TEXT,
    notes_path TEXT,
    git_commit TEXT,
    git_dirty INTEGER NOT NULL DEFAULT 0,
    cwd TEXT,
    command TEXT,
    manifest_path TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_experiment_events_type_family
    ON experiment_events(event_type, family);
CREATE INDEX IF NOT EXISTS idx_experiment_events_wandb
    ON experiment_events(wandb_run_id);
CREATE INDEX IF NOT EXISTS idx_experiment_events_status
    ON experiment_events(status);

CREATE TABLE IF NOT EXISTS experiment_event_artifacts (
    event_id TEXT NOT NULL REFERENCES experiment_events(event_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    direction TEXT NOT NULL DEFAULT 'unknown'
        CHECK(direction IN ('input', 'output', 'control', 'log', 'unknown')),
    path_or_uri TEXT NOT NULL,
    exists_local INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (event_id, role, path_or_uri)
);

CREATE INDEX IF NOT EXISTS idx_experiment_event_artifacts_path
    ON experiment_event_artifacts(path_or_uri);
CREATE INDEX IF NOT EXISTS idx_experiment_event_artifacts_role
    ON experiment_event_artifacts(role, artifact_type);

CREATE TABLE IF NOT EXISTS experiment_run_links (
    parent_run_id TEXT NOT NULL,
    child_run_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    event_id TEXT REFERENCES experiment_events(event_id) ON DELETE SET NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (parent_run_id, child_run_id, link_type)
);

CREATE INDEX IF NOT EXISTS idx_experiment_run_links_child
    ON experiment_run_links(child_run_id);

CREATE TABLE IF NOT EXISTS experiment_event_links (
    src_event_id TEXT NOT NULL REFERENCES experiment_events(event_id) ON DELETE CASCADE,
    dst_event_id TEXT NOT NULL REFERENCES experiment_events(event_id) ON DELETE CASCADE,
    link_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (src_event_id, dst_event_id, link_type)
);
