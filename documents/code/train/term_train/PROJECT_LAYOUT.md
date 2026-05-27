# Term Retriever Training Layout

This directory is being migrated from a flat experiment scratchpad to a tracked
control surface.  Do not move existing active launchers casually; Slurm logs and
WandB notes may still reference their current paths.

New files should use:

- `src/` for reusable Python code.
- `common/` for shared shell bodies such as the HN-depth common launcher.
- `launchers/YYYY/MM/` for concrete sbatch launchers.
- `notes/YYYY/MM/` for WandB notes files.
- `manifests/YYYY/MM/` for JSON event manifests.
- `reports/` for writeups and local analysis.
- `archive/` for retired launchers kept for reference.

Every new train/eval launcher needs a matching event manifest registered with:

```bash
python documents/code/general/experiment_event.py register <manifest.json>
```

Launch through the wrapper when possible:

```bash
python documents/code/general/experiment_event.py launch <manifest.json> -- \
  sbatch --parsable <launcher.sh>
```

The root-level files remain supported during migration.  When editing a root
launcher substantially, either create a new timestamped launcher under
`launchers/YYYY/MM/` or add/update a manifest in `manifests/YYYY/MM/`.

