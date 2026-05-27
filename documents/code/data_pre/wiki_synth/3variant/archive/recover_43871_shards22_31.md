# Recovery note: TTS job 43871 shards 22-31

Created: 2026-04-25

Context: teammate is running shards `22-31` on a separate machine. The local taurus job `43871` can be deprioritized for these shards, but this note preserves the exact recovery parameters in case the teammate run fails or is incomplete.

## Cancellation record

On 2026-04-25, local taurus shards `43871_[22-31]` were cancelled to free GPUs after teammate started running the same shard range.

Cancel command used:

```bash
scancel '43871_[22-31]'
```

Post-cancel `squeue -j 43871` showed only local shards `18`, `20`, and `21` still running. Shards `22-31` are expected to be supplied by teammate, or recovered with the command below.

## Current SLURM snapshot

Observed via `squeue -j 43871`:

```text
43871_[26-31]  PENDING  (Priority)
43871_25       RUNNING
43871_24       RUNNING
43871_23       RUNNING
43871_22       RUNNING
```

Other local shards were still part of the same array at the snapshot time, but the delegated/recovery range is only `22-31`.

## Original local job

- SLURM array job: `43871`
- Job name: `tts_gs_v2_full`
- Submit script:
  `/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/3variant/run_tts_3variant_gigaspeech_full_taurus.sh`
- Work dir: `/tmp`
- Logs:
  `/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_v2_full.out`
  `/mnt/gemini/data1/jiaxuanluo/logs/%A-%a_tts_gs_v2_full.err`

SLURM resources:

```bash
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --array=0-31
#SBATCH --chdir=/tmp
```

## Recovery command

If teammate output for any of `22-31` is missing/bad, rerun exactly those shard ids locally. Do not renumber them.

```bash
sbatch --array=22-31 \
  /home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/3variant/run_tts_3variant_gigaspeech_full_taurus.sh
```

For partial recovery, pass only missing shard ids, for example:

```bash
sbatch --array=22,25,29-31 \
  /home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/3variant/run_tts_3variant_gigaspeech_full_taurus.sh
```

Critical invariant: `--num-shards` must stay `32`, and `SLURM_ARRAY_TASK_ID` must be the real global shard id (`22` through `31`), not remapped to `0-9`.

## Exact TTS parameters

From `run_tts_3variant_gigaspeech_full_taurus.sh`:

```bash
COSYVOICE_ROOT="/mnt/gemini/home/jiaxuanluo/CosyVoice"
CONDA_ENV="/mnt/gemini/home/jiaxuanluo/miniconda3/envs/cosyvoice_vllm"

SCRIPT_PATH="/mnt/taurus/home/jiaxuanluo/InfiniSST/documents/code/data_pre/tts/different_variants/rag_tts_multispeaker_noise.py"
DATA_PATH="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_utterances_3variant.jsonl"
OUTPUT_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full"
MODEL_DIR="${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B"

SPEAKER_DIR="/mnt/gemini/home/jiaxuanluo/gigaspeech_speaker_prompts"
NOISE_DIR=""

TOTAL_SHARDS=32
BATCH_SIZE=16
SNR_LOW=5
SNR_HIGH=25
OUTPUT_JSONL_PREFIX="wiki_synth_3variant_gs_v2_clean_with_tts"
```

Python invocation for each shard:

```bash
python "${SCRIPT_PATH}" \
  --data "${DATA_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --speaker-dir "${SPEAKER_DIR}" \
  --noise-dir "" \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 32 \
  --batch-size 16 \
  --snr-low 5 \
  --snr-high 25 \
  --no_dedup \
  --output_jsonl_prefix "wiki_synth_3variant_gs_v2_clean_with_tts"
```

## Expected outputs

WAVs:

```text
/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full/clean/
```

Shard JSONLs are written next to the input JSONL, not under `OUTPUT_DIR`:

```text
/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_3variant_gs_v2_clean_with_tts_shard22.jsonl
...
/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_3variant_gs_v2_clean_with_tts_shard31.jsonl
```

## Downstream pipeline expectation

The monitor script expects all 32 shard JSONLs:

```text
/home/jiaxuanluo/InfiniSST/documents/code/data_pre/wiki_synth/3variant/monitor_and_run_pipeline_gsv2.sh
```

Relevant settings:

```bash
TTS_JOB_ID="${TTS_JOB_ID:-43871}"
TTS_WAV_DIR="/mnt/gemini/home/jiaxuanluo/wiki_synth_tts_3variant_gigaspeech_full"
TTS_JSONL_DIR="/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant"
TTS_SHARD_PREFIX="wiki_synth_3variant_gs_v2_clean_with_tts"
TOTAL_TTS_SHARDS=32
MERGED_JSONL="${TTS_WAV_DIR}/wiki_synth_3variant_gs_v2_clean_dual.jsonl"
```

Before merge, verify the recovered or teammate-returned shards exist:

```bash
for i in $(seq 22 31); do
  wc -l "/mnt/gemini/data1/jiaxuanluo/wiki_synth_data/3variant/wiki_synth_3variant_gs_v2_clean_with_tts_shard${i}.jsonl"
done
```
