# Retriever Training Terms

This directory holds reproducible preprocessing utilities for retriever training
term JSONL files.

## GigaSpeech MFA Event Dedup

GigaSpeech retriever audio is cut into 1.92s chunks with 0.96s stride. The
overlap is useful because phrase terms can cross the original 0.96s unit
boundary, but after MFA supervision the same absolute acoustic term event may
appear in two adjacent training rows.

For the deduped data line, we keep one random row per GigaSpeech absolute MFA
event and leave wiki-synth rows unchanged:

```bash
bash documents/code/data_pre/training_terms_for_retriever/run_dedup_gsv2full_gsrepaired.sh
```

The event key is:

```text
(utter_id, normalized_term_key, chunk_idx * 0.96 + mfa_term_start_in_chunk,
 chunk_idx * 0.96 + mfa_term_end_in_chunk)
```

The random choice uses a fixed seed (`20260509` by default), so rerunning the
script is deterministic unless `SEED` is overridden.

Default output:

```text
/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup.jsonl
```

The companion stats JSON records the input row counts, deduped GigaSpeech event
count, duplicate rows dropped, and final written row counts.

## 3.84s Context Expansion

The 3.84s context builder starts from the deduped 1.92s GigaSpeech rows and
cuts a longer chunk from the original GigaSpeech opus file using the MFA SQLite
index. For each longer speech group, it writes every known term event whose MFA
span overlaps the new window. This is intentionally conservative: terms newly
covered by the wider context become positives in the same group instead of
hard-negative false negatives.

`wiki_synth_` rows are also recut to real 3.84s audio. The builder infers the
original TTS WAV and wiki-synth TextGrid from each 1.92s chunk path, cuts a
longer chunk around the original window, and rewrites `chunk_src_text`,
`chunk_audio_path`, and `mfa_term_*_in_chunk` for the wider chunk.

```bash
bash documents/code/data_pre/training_terms_for_retriever/run_expand_gsv2full_gsdedup_gsctx3p84.sh
```

For faster rebuilds, use the utterance-hash sharded wrapper. This keeps every
row for the same `utter_id` in one shard, so GigaSpeech false-negative masking
still sees all term events for the wider window:

```bash
NUM_SHARDS=8 PARALLEL_JOBS=4 \
  bash documents/code/data_pre/training_terms_for_retriever/run_expand_gsv2full_gsdedup_gsctx3p84_parallel.sh
```

Default output:

```text
/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_gsctx3p84.jsonl
```

Default chunk WAV directory:

```text
/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_gsctx3p84
```

Default wiki-synth chunk WAV directory:

```text
/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_gsctx3p84/wiki_synth
```

## Variable Context Dataset

The variable-context builder creates a retriever training JSONL with 0.96s,
1.92s, 2.88s, and 3.84s chunks. The default assignment mode balances output
rows across the four buckets, targeting 25% per duration. GigaSpeech windows are
expanded from the MFA event inventory, and every known term event overlapping
the chosen speech window is written as a positive row. This keeps newly audible
terms from becoming false-negative terms.

`wiki_synth_` rows are recut from the inferred TTS WAV and TextGrid for the
chosen duration. If a rare row cannot be safely recut, it is kept as a 1.92s
fallback and marked with `context_expand_failure`.

```bash
NUM_SHARDS=8 PARALLEL_JOBS=4 \
  bash documents/code/data_pre/training_terms_for_retriever/run_build_gsv2full_gsdedup_varctx_0p96_1p92_2p88_3p84_parallel.sh
```

Default output:

```text
/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx0p96_1p92_2p88_3p84.jsonl
```

Default chunk WAV directory:

```text
/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx0p96_1p92_2p88_3p84
```

After building, run the diagnostic checker to verify duration balance, domain
mix, sampled audio frame counts, MFA span validity, and stats consistency:

```bash
bash documents/code/data_pre/training_terms_for_retriever/run_diagnose_gsv2full_gsdedup_varctx_0p96_1p92_2p88_3p84.sh
```

For inference-matched training with the fixed 1.92s look-back included in every
latency multiplier, use the look-back variant instead. Its four buckets are
2.88s, 3.84s, 4.80s, and 5.76s, corresponding to `lm=1..4` plus the 1.92s
look-back. The same MFA overlap rule is used, so newly covered GigaSpeech terms
are written as positives.

```bash
NUM_SHARDS=8 PARALLEL_JOBS=4 \
  bash documents/code/data_pre/training_terms_for_retriever/run_build_gsv2full_gsdedup_varctx_lmlb2p88_3p84_4p80_5p76_parallel.sh
```

Default output:

```text
/mnt/gemini/home/jiaxuanluo/term_train_3variant_gsv2_full0_31_clean_mfa_gsrepaired_gsdedup_varctx2p88_3p84_4p80_5p76.jsonl
```

Default chunk WAV directory:

```text
/mnt/gemini/home/jiaxuanluo/term_train_audio_chunks_gsv2full_gsdedup_varctx2p88_3p84_4p80_5p76
```

Diagnostic wrapper:

```bash
bash documents/code/data_pre/training_terms_for_retriever/run_diagnose_gsv2full_gsdedup_varctx_lmlb2p88_3p84_4p80_5p76.sh
```

## Eval JSONLs for 3.84s Context

The primary dev eval uses the existing latency-multiplier-4 dataset:

```text
/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_m4.jsonl
```

ACL extracted-paper eval can be regenerated with 3.84s chunks and 1.92s
stride from the cached MFA TextGrids and full WAV files:

```bash
bash documents/code/data_pre/paper_extracted/run_prepare_acl6060_extracted_ctx3p84.sh
```
