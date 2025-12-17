# Term-Level Training Data (180k Samples)

This note documents how the InfiniSST term-level training dataset is constructed and where to find the generated data.

## Dataset Summary

- Segment-level train file: `/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/term_training_180k.jsonl`
- Segment-level dev file: `/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/term_training_dev.jsonl`
- Speech-level train file: `/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/term_training_180k_speech.jsonl`
- Speech-level dev file: `/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/term_training_dev_speech.jsonl`
- Total segments (after carving out dev): `179,990`
  - Term-containing segments: `≈90,000`
  - `no_term` segments: `≈90,000`
- Segment-level instances (grouped by `PODxxxx_Syyyyyy` prefix): `124,447` train / `7` dev
- Speech-level instances (grouped by `PODxxxx` recording id): `12,272` train / `3` dev (still covering the same 179,990 / 10 segments respectively)
- Source chunks: `/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/samples/xl_cleaned/term_level_chunks_*.json`
- Glossary: `/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_cleaned.json`

For `no_term` segments the references block is `[]`. For term segments the references block contains the JSON objects derived from `term_chunk_audio_ground_truth_terms`. Each term’s translation comes from the glossary `target_translations["zh"]` when available; otherwise we fall back to the English term string so that every entry still has a non-empty translation field.

Each segment-level dataset file is emitted as **JSON Lines**, where each line aggregates all term chunks that share the same `PODxxxx_Syyyyyy` prefix. A single instance expands to:

```
{
  "messages": [
    {"role": "system", "content": "You are a professional simultaneous interpreter..."},
    {"role": "user", "content": "<audio>, references: {...}"},
    {"role": "assistant", "content": "<term_chunk_text>"},
    {"role": "user", "content": "<audio>, references: []"},
    {"role": "assistant", "content": "<term_chunk_text>"}
  ],
  "audios": [
    "/mnt/.../POD0000000002_S0000015_term_....wav",
    "/mnt/.../POD0000000002_S0000015_term_no_term_....wav"
  ]
}
```

Thus each instance corresponds to one original speech segment and contains multiple user/assistant turns as needed (system prompt repeated per instance). We no longer emit the `instance_id` field, but the grouping order still follows the chronological chunk order from the source files. The speech-level JSON files further merge every segment from the same recording (e.g., `POD0000000003`) into one instance by concatenating their turns and audio paths.

## Generation Script

The reusable builder script lives at `retriever/gigaspeech/build_term_training_dataset.py`. It streams the cleaned chunk files, balances term/no-term samples, fetches glossary translations, and writes a flat list of chunk-level entries. After generation, run `retriever/gigaspeech/reshape_term_datasets.py` to group chunk entries into per-prefix instances. The key commands are:

```
# 1) Build chunk-level dataset (flat list)
python retriever/gigaspeech/build_term_training_dataset.py \
  --target-samples 180000 \
  --output-path retriever/gigaspeech/data/samples/term_training_180k.jsonl \
  --output-format json \
  --term-oversample-factor 6 \
  --chunk-dir retriever/gigaspeech/data/samples/xl_cleaned \
  --glossary-path retriever/gigaspeech/data/terms/glossary_cleaned.json \
  --no-term-ratio 0.5 \
  --target-lang zh

# 2) Reshape train & dev into per-segment JSONL
python retriever/gigaspeech/reshape_term_datasets.py \
  --output-format jsonl \
  retriever/gigaspeech/data/samples/term_training_180k.jsonl \
  retriever/gigaspeech/data/samples/term_training_dev.jsonl

# 3) (Optional) Create speech-level aggregations
cp retriever/gigaspeech/data/samples/term_training_180k.jsonl \
   retriever/gigaspeech/data/samples/term_training_180k_speech.jsonl
cp retriever/gigaspeech/data/samples/term_training_dev.jsonl \
   retriever/gigaspeech/data/samples/term_training_dev_speech.jsonl
python retriever/gigaspeech/reshape_term_datasets.py \
  --output-format jsonl \
  --group-level speech \
  retriever/gigaspeech/data/samples/term_training_180k_speech.jsonl \
  retriever/gigaspeech/data/samples/term_training_dev_speech.jsonl
```

Pass `--output-format jsonl` if you prefer newline-delimited intermediate records. The builder logs how many chunk entries were scanned, how many unique terms were resolved (including those that fallback to English), and confirms the final 50/50 split before reshaping. Use `reshape_term_datasets.py --drop-instance-id-only ...` if you need to strip legacy `instance_id` keys from already reshaped files.

### Parameters

- `--target-samples`: Total number of records to emit.
- `--no-term-ratio`: Portion of samples that should come from `no_term` segments (default `0.5`).
- `--term-oversample-factor`: Multiplier for how many candidate term samples to gather before filtering by translation availability. Increase this value if you need more coverage for rare terms.
- `--max-files`: Optional limit if you want to constrain chunk scanning during debugging.
- `--output-format`: `json` (default, emits a single mega conversation) or `jsonl`.

## Validation

- `python - <<'PY'` / `len(json.load(open(...)))` → `124447` segment-level instances for train, `7` for dev; `12272` speech-level instances for train, `3` for dev
- Manual spot checks confirm that each instance groups all term chunks sharing the same prefix and that `audios[i]` aligns with the `(i+1)`-th user turn inside that instance.
- Logs emitted by the builder record:
  - Total processed chunk entries: `724,361`
  - Unique terms covered: `38,993` (with ~61.5% of them falling back to the English term text because no zh translation was available).
  - Total retained segments after reshaping: `179,990` train / `10` dev

This dataset covers roughly 100 hours of audio (≈180k segments) and keeps the required `no_term` proportion at exactly 50%. Use the builder + reshape scripts to regenerate or adjust the balance if future requirements change.

