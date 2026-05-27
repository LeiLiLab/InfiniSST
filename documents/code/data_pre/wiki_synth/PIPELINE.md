# Wiki-Synth Data Pipeline

This directory keeps the current wiki-synth data builders plus an `archive/`
for older one-off launchers. The current retriever data line is the
GigaSpeech-voice-pool v2, clean-only, 3-variant wiki synthesis pipeline.

## Current Active Flow

1. Build GigaSpeech voice prompts for CosyVoice:

   ```bash
   sbatch documents/code/data_pre/wiki_synth/run_build_gigaspeech_voice_pool_v2.sh
   ```

2. Prepare/synthesize 3-variant wiki utterances:

   ```bash
   sbatch documents/code/data_pre/wiki_synth/3variant/run_gemini_3variant.sh
   sbatch --array=0-31 documents/code/data_pre/wiki_synth/3variant/run_tts_3variant_gigaspeech_full_taurus.sh
   ```

3. Merge TTS output and run MFA/chunk cutting:

   ```bash
   python documents/code/data_pre/wiki_synth/3variant/merge_gsv2_full0_31.py
   sbatch --array=0-31 documents/code/data_pre/wiki_synth/3variant/run_mfa_3variant_gsv2_full0_31_taurus.sh
   ```

4. Build the retriever training JSONL and repair GigaSpeech MFA spans:

   ```bash
   sbatch documents/code/data_pre/wiki_synth/3variant/run_build_train_gsv2_full0_31_taurus.sh
   sbatch documents/code/data_pre/wiki_synth/3variant/run_repair_gigaspeech_mfa_spans_gsv2_full0_31_taurus.sh
   ```

5. For the 3.84s context ablation, expand only the real GigaSpeech rows:

   ```bash
   bash documents/code/data_pre/training_terms_for_retriever/run_expand_gsv2full_gsdedup_gsctx3p84.sh
   ```

## 3.84s Context Policy

The 3.84s context expansion script uses GigaSpeech MFA TextGrids and original
opus audio to recut longer real-speech chunks. It also recuts `wiki_synth_`
rows from the original TTS WAV and wiki-synth TextGrid inferred from each
existing 1.92s chunk path:

```text
.../MFA/3variant_gsv2_*/chunks/shard_xx/utt_N_clean.wav
  -> .../MFA/3variant_gsv2_*/work/shard_xx/mfa_input/utt_N.wav
  -> .../MFA/3variant_gsv2_*/work/shard_xx/mfa_output/utt_N.TextGrid
```

The recut row rewrites `chunk_audio_path`, `chunk_src_text`,
`mfa_term_start_in_chunk`, and `mfa_term_end_in_chunk` for the 3.84s audio.

## Active Scripts

Root directory:

- `align_and_cut_wiki_synth.py` - MFA alignment and wiki TTS chunk cutting.
- `build_gigaspeech_voice_pool.py` - build GigaSpeech voice-reference prompts.
- `run_build_gigaspeech_voice_pool_v2.sh` - current voice-pool launcher.
- `build_p31_ranked_terms.py`, `build_leftover_train_terms.py` - P31 term
  ranking/filtering utilities.
- `build_untrained_p31_dev_glossary.py`,
  `build_translated_untrained_p31_glossary.py` - eval glossary builders.
- `extract_rdf_terms_with_p31.py`, `sample_wiki_terms_by_domain.py`,
  `generate_term_utterances.py` and their current wrappers.

`3variant/`:

- `prepare_top1m_terms.py`
- `merge_tts_3variant.py`
- `merge_gsv2_full0_31.py`
- `build_train_3variant.py`
- `run_gemini_3variant.sh`
- `run_tts_3variant_gigaspeech_full_taurus.sh`
- `run_mfa_3variant_gsv2_full0_31_taurus.sh`
- `run_build_train_gsv2_full0_31.sh`
- `run_build_train_gsv2_full0_31_taurus.sh`
- `run_repair_gigaspeech_mfa_spans_gsv2_full0_31_taurus.sh`

## Archive

Historical POC launchers, hard-coded monitor scripts, partial0-20/extra21-31
recovery scripts, teammate handoff files, and old v1 train builders live under:

- `documents/code/data_pre/wiki_synth/archive/`
- `documents/code/data_pre/wiki_synth/3variant/archive/`
