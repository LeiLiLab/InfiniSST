#!/bin/bash

# SONAR完整评估脚本（srun 版，非交互）
# 参数: $1 = n, $2 = text_field, $3 = single_slice, $4 = test_samples_path

set -euo pipefail

# --- Params ---
n=${1:-2}
text_field=${2:-term}
single_slice=${3:-true}
test_samples_path=${4:-"data/samples/xl/term_level_chunks_500000_1000000.json"}

# --- Model path & job name ---
model_save_path="data/clap_sonar_term_level_single.pt"
job_name="sonar_eval"

if [[ "$single_slice" == "true" ]]; then
  model_save_path="data/clap_sonar_term_level_single_best.pt"
  job_name="sonar_train_term_level_single"
else
  model_save_path="data/clap_sonar_term_level_full_best.pt"
  job_name="sonar_train_term_level_full"
fi

# --- Logs ---
mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
out_log="logs/${job_name}_${ts}.out"
err_log="logs/${job_name}_${ts}.err"

# --- Command to run inside allocation ---
inner_cmd=$(cat <<'EOF'
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_full_evaluate.py \
  --model_path='"$model_save_path"' \
  --build_offline_assets \
  --asset_out_dir data \
  --index_type ivfpq \
  --use_ip \
  --nlist 4096 \
  --pq_m 64 \
  --pq_bits 8 \
  --nprobe 16 \
  --test_samples_path='"$test_samples_path"' \
  --max_eval=1000
EOF
)

# Note:
# - srun here requests a fresh allocation and runs non-interactively.
# - Adjust --time as your policy allows; add --qos if your cluster uses it.

echo "Submitting with srun… logs: $out_log / $err_log"

srun \
  --job-name="$job_name" \
  --partition=taurus \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=16 \
  --gres=gpu:2 \
  --mem=96G \
  --time=24:00:00 \
  --output="$out_log" \
  --error="$err_log" \
  -u bash -lc "$inner_cmd"

exit_code=$?
echo "srun finished with exit code: $exit_code"
echo "Stdout: $out_log"
echo "Stderr: $err_log"
exit $exit_code