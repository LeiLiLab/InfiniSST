# InfiniSST Agent Experiment Control Rules

This document is the agent-facing operating rule for experiment management in
`/home/jiaxuanluo/InfiniSST`.  It replaces ad hoc notes-only tracking with a
lineage-first workflow: WandB stores run truth, event manifests store
provenance, and SQLite is a generated index for fast lookup.

## 1. Core Principle

Every experiment action must answer these questions without guessing:

- Which launcher and command created this run or artifact?
- Which data, glossary, checkpoint, and config files were consumed?
- Which checkpoint, eval output, report, or log files were produced?
- Which WandB run id and Slurm job id correspond to the action?
- Which prior runs or data-prep events does this event depend on?

If the answer is not in WandB or an event manifest, provenance is incomplete.
Do not reconstruct it from memory unless explicitly labelled as a backfill based
on evidence.

## 2. Source Of Truth

WandB is authoritative for:

- run config
- run metrics
- code snapshot and diff
- run status
- verdict
- final metric bundles

Event manifests are authoritative for:

- launcher path
- launch command
- Slurm job id
- WandB run id link
- notes path
- input artifacts
- output artifacts
- logs and reports
- parent or baseline run links
- downstream event links

Git is authoritative for source code.

SQLite at `documents/code/.cache/experiments.sqlite` is only a rebuildable index
derived from WandB plus event manifests.  If SQLite disagrees with WandB,
WandB wins for metrics/config/status.  If SQLite disagrees with a manifest,
the manifest should be corrected and re-registered.

Do not maintain hand-written metric tables as truth.  Reports can summarize
results, but metric values must remain in WandB and can only be cached by
`db-sync`.

## 3. Required Tools

Use the repo tools:

```bash
python documents/code/general/wandb_tool.py ...
python documents/code/general/experiment_event.py ...
```

Do not hand-roll `wandb.Api()` snippets for read-side queries unless
`wandb_tool.py` cannot express the query.  The CLI encodes project conventions,
metric presets, at-best-step reporting, and SQLite sync.

Useful event commands:

```bash
python documents/code/general/experiment_event.py register <manifest.json>
python documents/code/general/experiment_event.py launch <manifest.json> -- sbatch --parsable <launcher.sh>
python documents/code/general/experiment_event.py find --family <family>
python documents/code/general/experiment_event.py show <event_id-or-run_id>
python documents/code/general/experiment_event.py files <event_id-or-run_id>
```

Useful WandB commands:

```bash
python documents/code/general/wandb_tool.py --project <project> db-sync --runs <run_id>
python documents/code/general/wandb_tool.py --project <project> db-show <run_id> --config --notes
python documents/code/general/wandb_tool.py --project <project> compare <run_ids...> --preset retriever_eval retriever_train --at-best-step --anchor-metric both
python documents/code/general/wandb_tool.py --project <project> db-compare <run_ids...> --refresh --anchor-metric both
```

## 4. Event Manifest Is Mandatory

Before any data-prep, retriever training, retriever eval, speech LLM training,
offline eval, simuleval, stream eval, or analysis event, create and register a
manifest under:

```text
documents/code/<module>/manifests/YYYY/MM/<event_id>.json
```

Use the template:

```text
documents/code/_templates/experiment_event_manifest_template.json
```

Each manifest must include:

```text
schema_version
event_id
event_type
family
variant
status
project
wandb_run_id
slurm_job_id
launcher_path
notes_path
cwd
command
parent_run_ids
artifacts
metadata
```

Allowed event types:

```text
data_prepare
retriever_train
retriever_eval
speech_llm_train
offline_eval
simuleval
stream_eval
analysis
maintenance
```

Every artifact entry must include:

```text
role
type
direction
path
metadata
```

Directions:

```text
input
output
control
log
unknown
```

Common artifact roles:

```text
launcher
common_launcher
train_script
eval_script
notes
train_jsonl
dev_jsonl
acl_dev_jsonl
medicine_dev_jsonl
eval_wiki_glossary
acl_eval_wiki_glossary
medicine_eval_wiki_glossary
train_exclude_term_glossary
checkpoint
checkpoint_dir
index_dir
eval_output
report
slurm_logs
```

Notes files explain intent and verdict.  Manifests link exact files.  Do not
rely on notes alone for provenance.

## 5. File Layout For New Work

Do not create a new folder for every ablation.  Use a fixed module structure
and timestamped files.

For `documents/code/train/term_train/`:

```text
src/
common/
launchers/YYYY/MM/
notes/YYYY/MM/
manifests/YYYY/MM/
reports/
archive/
```

Use the same shape for:

```text
documents/code/simuleval/
documents/code/offline_evaluation/
documents/code/data_pre/<module>/
```

Existing root-level files may stay during migration.  New experiments should
use the fixed subfolders.  When substantially editing an old root-level launcher,
either create a new timestamped launcher under `launchers/YYYY/MM/` or add/update
a manifest under `manifests/YYYY/MM/`.

Do not put dependency folders such as `node_modules`, local WandB run folders,
logs, cache directories, or generated frontend build outputs under experiment
data/control roots unless there is a specific reason.  If they already exist,
ignore them for provenance unless explicitly referenced as artifacts.

## 6. Pre-Launch Workflow

Before launching any training or eval:

1. Query WandB for candidate baselines with `wandb_tool.py`.
2. If the user did not name a baseline, report top candidates and ask once.
3. Never quote baseline metrics from memory, markdown, chat history, or stale
   SQLite rows.
4. Verify the notes file has all required sections.
5. Verify WandB tags after prefixes are `1..64` chars.
6. Create and register an event manifest.
7. Prefer launching through `experiment_event.py launch`.
8. Watch startup logs until WandB init succeeds or a run id appears.
9. If WandB init fails, cancel the Slurm job.  Do not let untracked train/eval
   continue.
10. Once the WandB run id appears, add it to the manifest, re-register the
    manifest, and sync WandB into SQLite.

WandB sync after init:

```bash
python documents/code/general/wandb_tool.py --project <project> db-sync --runs <run_id>
```

Launch wrapper:

```bash
python documents/code/general/experiment_event.py launch <manifest.json> -- \
  sbatch --parsable <launcher.sh>
```

## 7. Required WandB Schema

Every training/eval script should initialize WandB with:

```python
wandb.init(
    project="<qwen3_rag|sst_omni|simuleval_eval>",
    name=f"{family}__{variant}__{YYYYMMDD-HHMM}__{key_hp}",
    config=vars(args),
    tags=[
        f"family:{family}",
        f"task:{task}",
        f"data:{data_tag}",
        "status:running",
    ],
    notes=open(args.notes_file).read(),
    save_code=True,
)
wandb.config.update({"baseline_run_ids": args.baseline_run_ids})
```

At run end:

- set `run.summary["verdict"]`
- replace `status:running` with `status:success` or `status:failed`
- for eval runs, set `run.config["trained_from_run"]`
- sync the run with best bundles

## 8. Required Notes Sections

Every notes file must include non-empty:

```text
## Hypothesis
## Background / Motivation
## What changed vs baseline
## Expected metrics
## Verdict
```

Notes are not a metrics database.  Do not hand-enter final metric values into
notes as the authoritative record.  Metrics live in WandB.

After a run finishes:

1. Fill `## Verdict`.
2. Push verdict to WandB notes/summary.
3. Flip the `status:*` tag to `success` or `failed`.
4. Refresh SQLite:

```bash
python documents/code/general/wandb_tool.py --project <project> db-sync --runs <run_id> --best-bundles
```

## 9. Metric Reporting Discipline

Cross-run metric tables must use at-best-step reads of WandB history, not raw
`run.summary[...]` loop-logged values.

Use:

```bash
python documents/code/general/wandb_tool.py --project <project> compare <run_ids...> \
  --preset retriever_eval retriever_train \
  --at-best-step \
  --anchor-metric both
```

or:

```bash
python documents/code/general/wandb_tool.py --project <project> db-compare <run_ids...> \
  --refresh \
  --anchor-metric both
```

Every cross-run table must include both bundles:

```text
primary   = best/step
secondary = best_secondary/step
```

Allowed direct summary keys:

```text
best/metric_value
best/step
best_secondary/metric_value
best_secondary/step
verdict
verdict_metrics
```

Forbidden direct summary keys:

```text
eval_acl6060/*
eval_dev/*
by_paper/*
train/*
train/tcm_pos_viol_rate
train/tcm_neg_viol_rate
train/loss*
```

Rule of thumb: if a key is logged inside the training loop, raw
`run.summary[key]` is a last-step snapshot, not a best-checkpoint metric.

Running runs may appear in comparisons only if explicitly labelled as running
and not final.

## 10. Data Leakage And Calibration Constraints

ACL is held-out test/readout data.  Do not use ACL to choose:

- tau
- model checkpoint
- threshold
- hyperparameters
- variant winner

For tau/model selection:

1. Use dev calibration first.
2. Freeze the selection rule.
3. Report ACL only as held-out readout.

If recall retention matters, use explicit recall-retention constraints rather
than F-score-only ranking.

When the user asks for reviewer-defensible calibration, state the rule before
showing held-out results.

## 11. Query And Backfill Behavior

When asked:

```text
Which script/data/eval belongs to run X?
Which evals descend from model X?
What files were used for this run?
```

First query:

```bash
python documents/code/general/experiment_event.py files <run_id-or-event_id>
python documents/code/general/experiment_event.py show <run_id-or-event_id>
python documents/code/general/wandb_tool.py --project <project> db-show <run_id> --config --notes
```

If no manifest exists, say provenance is incomplete.  Then backfill a manifest
from evidence:

- WandB config
- WandB notes
- launcher references
- Slurm logs
- filesystem paths
- Git history when needed

Label backfilled facts as evidence-derived.  Do not present guesses as verified
lineage.

## 12. Human And Agent Division Of Labor

The human owns:

- canonical run decisions
- deprecated run decisions
- reviewer-facing protocol
- experiment family taxonomy
- final judgment about which results matter

The agent owns:

- manifest creation and updates
- launcher hash capture
- command capture
- Slurm job id capture
- WandB run id linking
- artifact path extraction
- SQLite registration and sync
- lineage queries
- orphan launcher and missing-manifest audits

The agent should not freely create new directory trees.  New files should go
under fixed module folders:

```text
launchers/YYYY/MM/
notes/YYYY/MM/
manifests/YYYY/MM/
reports/
archive/
```

## 13. Legacy Runs

A run is historical debt if it has no WandB record or lacks required schema.
Do not use historical-debt runs as authoritative baselines unless the user
explicitly asks for a legacy comparison and the limitations are stated.

For legacy recovery:

1. Read WandB `git_commit` and config if available.
2. Locate likely launcher and logs from filesystem/Git evidence.
3. Create a backfill event manifest.
4. Register the manifest.
5. Sync WandB if a run id exists.
6. Mark unresolved fields as `null` or `unknown`, not guessed.

## 14. RAG Policy

Do not build or trust RAG as the first layer of experiment tracking.

RAG may sit on top of the structured ledger, but answers must return stable
identifiers and paths:

```text
event_id
wandb_run_id
launcher_path
artifact path
manifest path
```

If RAG retrieves a plausible script without a manifest or DB link, treat it as
a candidate only and verify before use.

## 15. Practical Defaults

For new retriever training under `documents/code/train/term_train/`:

- create notes under `notes/YYYY/MM/`
- create launcher under `launchers/YYYY/MM/`
- create manifest under `manifests/YYYY/MM/`
- include `qwen3_glossary_neg_train.py` as `train_script`
- include common launcher body as `common_launcher` when sourced
- include all train/dev/ACL/medicine JSONL files as artifacts
- include all eval glossaries as artifacts
- include output checkpoint dir and Slurm log dir
- register manifest before launch
- re-register manifest after WandB run id appears

For new data-prep under `documents/code/data_pre/<module>/`:

- include raw input dirs/files
- include source scripts
- include output JSONL/glossary/audio/index dirs
- include any filler or exclusion glossary
- include downstream training manifest links if known

## 16. Hard Do Not Rules

Do not:

- quote final metrics from chat memory
- quote cross-run metrics from raw `run.summary[...]`
- hand-write metric truth into markdown
- manually edit SQLite metric tables
- launch train/eval without WandB
- let a train/eval continue after WandB init failure
- create a new folder for each ablation
- rely on notes alone for provenance
- use ACL to select tau/checkpoint/hyperparameters
- guess which launcher produced a run
- mix data paths from similar launchers without manifest evidence

## 17. Minimal Checklist Before Answering Experiment Questions

Before answering a factual experiment question:

1. Identify whether the question is about metrics, provenance, or interpretation.
2. For metrics, use WandB CLI at-best-step reads.
3. For provenance, use event manifests and `experiment_event.py`.
4. For interpretation, cite the rule or manifest evidence used.
5. If evidence is missing, say so and propose a backfill.

