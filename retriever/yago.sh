#!/bin/bash

mapfile -t FILES < <(find data/yago -type f | sort)

MAX_ARRAY_SIZE=1000
TOTAL=${#FILES[@]}
echo "[INFO] Total input files: $TOTAL"

for ((i = 0; i < TOTAL; i += MAX_ARRAY_SIZE + 1)); do
    offset=$i
    remaining=$((TOTAL - i))
    if [ $remaining -gt $((MAX_ARRAY_SIZE + 1)) ]; then
        size=$((MAX_ARRAY_SIZE + 1))
    else
        size=$remaining
    fi

    echo "[INFO] Submitting batch: offset=$offset size=$size"
    sbatch --export=OFFSET=$offset --array=0-$((size - 1))%50 run_yago_batch.slurm
done