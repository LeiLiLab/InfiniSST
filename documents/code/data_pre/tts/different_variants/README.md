# Multi-Speaker + Noise Augmented TTS Pipeline

## Background

The original TTS pipeline (`rag_tts_wiki_synth.py`) uses CosyVoice with a **single fixed speaker** and generates **clean audio** (no background noise). This leads to overly homogeneous training data for the retriever — all synthesized speech sounds identical in timbre and acoustic conditions, which hurts retriever robustness on real-world audio.

This upgraded pipeline (`rag_tts_multispeaker_noise.py`) addresses both issues:

1. **Multi-speaker voice cloning**: randomly assigns one of 110 VCTK speakers per utterance via CosyVoice zero-shot inference, producing diverse timbres and accents.
2. **Noise augmentation**: randomly mixes in real-world background noise from WHAM! dataset at varying SNR levels.

## What Changed vs. Original

| Aspect | Original (`rag_tts_wiki_synth.py`) | Upgraded (`rag_tts_multispeaker_noise.py`) |
|---|---|---|
| TTS mode | `inference_zero_shot` with 1 fixed ref audio | `inference_zero_shot` with random speaker per utterance |
| Speaker diversity | 1 speaker | 110 speakers (VCTK), 13 accents, M/F |
| Reference text | Fixed placeholder (`"希望你以后能够做的比我还好呦。"`) | Per-speaker ground truth from `speaker_index.json` |
| Noise | None (clean audio) | WHAM! noise mixed at random SNR |
| Noise control | N/A | Configurable probability, SNR range |

## Data Dependencies

### Speaker Prompts

- **Path**: `/mnt/data/siqiouyang/datasets/vctk_speaker_prompts/`
- **Contents**: 110 WAV files (`p225.wav` .. `s5.wav`) + `speaker_index.json`
- **Source**: VCTK corpus, prepared by Siqi
- **Format**: 48kHz mono WAV, ~2-3s each
- **Metadata** (`speaker_index.json`): per-speaker `text`, `gender`, `accent`
- **Accents**: American, Australian, British, Canadian, English, Indian, Irish, NewZealand, NorthernIrish, Scottish, SouthAfrican, Welsh
- **Gender split**: ~50/50 M/F

### Noise Clips

- **Path**: `/mnt/data/siqiouyang/datasets/wham_wav/`
- **Contents**: 25,000 WAV files (`00000.wav` .. `24999.wav`)
- **Source**: WHAM! dataset (real-world ambient noise)
- **Format**: 16kHz mono WAV, 5-14s each

### Input Data

- **Path**: `/mnt/data/siqiouyang/datasets/gigaspeech/wiki_synth_utterances_1M_unique_1third_with_tts.jsonl`
- **Contents**: 333,327 entries, each with `term`, `utterance`, `variant_idx`
- **Utterance length**: mean 53.6 chars (range 13-137)

## How It Works

### Per-Utterance Processing

For each utterance indexed by `line_idx`:

1. **Speaker selection**: `random.Random(line_idx)` picks a VCTK speaker deterministically (reproducible across reruns and shards).
2. **TTS generation**: CosyVoice `inference_zero_shot` clones the selected speaker's voice to synthesize the utterance. The per-speaker `ref_text` from `speaker_index.json` is used for alignment.
3. **Noise mixing** (with probability `noise_prob=0.7`):
   - Pick a random WHAM! noise clip
   - Sample SNR uniformly from `[snr_low, snr_high]` dB (default `[5, 25]`)
   - Compute noise gain to match target SNR, mix into clean TTS audio
   - Normalize peak to prevent clipping

### Key Design Decisions

- **Deterministic randomness**: speaker and noise selection are seeded by `line_idx`, so the same utterance always gets the same augmentation regardless of shard assignment.
- **Skip-on-exist**: already generated WAV files are skipped, enabling safe resume after interruption.
- **Concurrent batching**: `--batch-size N` runs N parallel vLLM inference threads to maximize GPU utilization.

## Files

| File | Description |
|---|---|
| `rag_tts_multispeaker_noise.py` | Main Python script with multi-speaker + noise logic |
| `rag_tts_wiki_synth.py` | Original single-speaker script (kept for reference) |
| `tts_wiki_synth.sh` | SLURM array job script (8 shards) |
| `slurm_logs/` | SLURM stdout/stderr logs |

## Running

### SLURM Submission

```bash
cd /home/jiaxuanluo/InfiniSST/documents/code/data_pre/tts/different_variants
sbatch tts_wiki_synth.sh
```

This submits 8 parallel array tasks (one per GPU), each processing ~41,666 utterances.

### Manual Single-Shard Run

```bash
env CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  /mnt/gemini/home/jiaxuanluo/miniconda3/envs/cosyvoice_vllm/bin/python \
  rag_tts_multispeaker_noise.py \
    --data /mnt/data/siqiouyang/datasets/gigaspeech/wiki_synth_utterances_1M_unique_1third_with_tts.jsonl \
    --output-dir /mnt/data/jiaxuanluo/wiki_synth_utterances_tts_augmented \
    --speaker-dir /mnt/data/siqiouyang/datasets/vctk_speaker_prompts \
    --noise-dir /mnt/data/siqiouyang/datasets/wham_wav \
    --shard-id 0 --num-shards 8 --batch-size 4
```

### Disable Multi-Speaker or Noise

```bash
# Single fixed speaker (no VCTK), still with noise:
python rag_tts_multispeaker_noise.py --speaker-dir "" ...

# Multi-speaker, no noise:
python rag_tts_multispeaker_noise.py --noise-dir "" ...

# Both off (equivalent to original script):
python rag_tts_multispeaker_noise.py --speaker-dir "" --noise-dir "" ...
```

## Configuration Defaults

| Parameter | Default | Description |
|---|---|---|
| `--speaker-dir` | `/mnt/data/siqiouyang/datasets/vctk_speaker_prompts` | VCTK speaker prompt directory |
| `--noise-dir` | `/mnt/data/siqiouyang/datasets/wham_wav` | WHAM! noise directory |
| `--noise-prob` | 0.7 | Probability of adding noise per utterance |
| `--snr-low` | 5 dB | Minimum SNR for noise mixing |
| `--snr-high` | 25 dB | Maximum SNR for noise mixing |
| `--batch-size` | 4 | Concurrent vLLM requests per GPU |
| `--num-shards` | 8 | Total parallel shards |
| `--sampling-rate` | 16000 | Output audio sample rate |

## Output

- **Audio**: `/mnt/data/jiaxuanluo/wiki_synth_utterances_tts_augmented/{chunk_dir}/{line_idx}.wav`
- **JSONL** (per shard): `wiki_synth_utterances_1M_all_with_tts_shard{N}.jsonl` in the same directory as input data

## Estimated Runtime

- ~3.8s per utterance (smoke test on A6000, batch_size=1)
- 333,327 utterances / 8 shards ≈ 41,666 per shard
- Per shard: ~44 hours
- With batch_size=4: expected ~12-15 hours per shard
