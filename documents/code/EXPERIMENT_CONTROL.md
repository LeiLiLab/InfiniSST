# Experiment Control Layout

This repo uses three separate records for experiments:

- WandB is authoritative for run config, metrics, code snapshot, status, and verdict.
- Git is authoritative for source code.
- `documents/code/.cache/experiments.sqlite` is a rebuildable index for runs,
  event manifests, launchers, data paths, and downstream links.

Do not use notes files as the only provenance record.  Notes explain intent and
verdict; event manifests link the exact launcher, command, inputs, outputs, and
run ids.

## New File Layout

For new work, keep module roots small and put new files under fixed subfolders:

```text
documents/code/train/term_train/
  src/                    # reusable training/eval Python code
  common/                 # shared launcher bodies and shell helpers
  launchers/YYYY/MM/      # concrete sbatch or shell launchers
  notes/YYYY/MM/          # run notes referenced by WandB
  manifests/YYYY/MM/      # event manifests registered in SQLite
  reports/                # human analysis, no authoritative metrics
  archive/                # retired local scripts kept for reference
```

Use the same shape for `documents/code/simuleval/` and each high-churn
`documents/code/data_pre/<module>/` folder.  Existing root-level files may stay
where they are until touched; new experiments should use the fixed subfolders.

## Required Event Flow

Before launching any data-prep, training, eval, simuleval, or analysis event:

1. Create a JSON manifest from
   `documents/code/_templates/experiment_event_manifest_template.json`.
2. Register it:

   ```bash
   python documents/code/general/experiment_event.py register path/to/manifest.json
   ```

3. Launch through the wrapper when possible:

   ```bash
   python documents/code/general/experiment_event.py launch path/to/manifest.json -- \
     sbatch --parsable path/to/launcher.sh
   ```

4. After WandB init succeeds, add the WandB run id to the manifest and register
   again.  Then run the existing WandB sync:

   ```bash
   python documents/code/general/wandb_tool.py --project qwen3_rag db-sync --runs <run_id>
   ```

Useful queries:

```bash
python documents/code/general/experiment_event.py find --family sst_ood_hardneg
python documents/code/general/experiment_event.py show <event_id-or-run_id>
python documents/code/general/experiment_event.py files <event_id-or-run_id>
```

