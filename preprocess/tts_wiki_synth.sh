#!/usr/bin/env bash

##SBATCH --nodelist=babel-4-23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:L40S:1
#SBATCH --partition=general
#SBATCH --time=2-00:00:00
##SBATCH --dependency=afterok:job_id
#SBATCH --array=0-5
##SBATCH --account=siqiouya
#SBATCH --mail-type=ALL
#SBATCH --mail-user=siqiouya@andrew.cmu.edu
#SBATCH -e slurm_logs/%A-%a.err
#SBATCH -o slurm_logs/%A-%a.out

source /home/siqiouya/miniconda3/bin/activate cosyvoice_vllm

echo "Shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} - TTS for wiki synthetic utterances"

python /home/siqiouya/code/infinisst-omni/preprocess/rag_tts_wiki_synth.py \
    --data /data/group_data/li_lab/siqiouya/datasets/gigaspeech/wiki_synth_utterances_leftover_1third.jsonl \
    --output-dir /compute/babel-p5-24/siqiouya/wiki_synth_utterances_leftover_tts_multispk \
    --speaker-index /data/group_data/li_lab/siqiouya/datasets/vctk_speaker_prompts/speaker_index.json \
    --noise-dir /data/group_data/li_lab/siqiouya/datasets/wham_wav \
    --shard-id $SLURM_ARRAY_TASK_ID \
    --num-shards $SLURM_ARRAY_TASK_COUNT