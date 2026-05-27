# Training-Term Data Preparation Layout

This module prepares retriever train/dev/eval JSONL and glossary artifacts.
New data-prep events should have manifests because their outputs are later
consumed by retriever training and offline eval.

Use:

- `src/` for reusable Python code.
- `launchers/YYYY/MM/` for concrete shell launchers.
- `manifests/YYYY/MM/` for event manifests.
- `reports/` for diagnostics and summaries.
- `archive/` for retired one-off scripts.

Register every new data-prep event:

```bash
python documents/code/general/experiment_event.py register <manifest.json>
```

