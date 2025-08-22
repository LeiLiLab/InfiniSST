#!/bin/bash

# SONAR完整评估脚本（srun 版，非交互）
# 支持原有评估模式和新的ACL评估模式
# 参数: $1 = n, $2 = text_field, $3 = single_slice, $4 = test_samples_path, $5 = acl_mode

set -euo pipefail

# --- Params ---
n=${1:-2}
text_field=${2:-term}
single_slice=${3:-true}
test_samples_path=${4:-"data/samples/xl/term_level_chunks_500000_1000000.json"}
acl_mode=${5:-false}  # 新增ACL模式开关

# --- Model path & job name ---
model_save_path="data/clap_sonar_term_level_single.pt"
job_name="sonar_eval"

if [[ "$single_slice" == "true" ]]; then
  model_save_path="data/clap_sonar_term_level_single_best.pt"
  job_name="sonar_eval_single"
else
  model_save_path="data/clap_sonar_term_level_full_best.pt"
  job_name="sonar_eval_full"
fi

# ACL模式调整
if [[ "$acl_mode" == "true" ]]; then
  job_name="${job_name}_acl"
fi

# --- Logs ---
mkdir -p logs
ts=$(date +%Y%m%d_%H%M%S)
out_log="logs/${job_name}_${ts}.out"
err_log="logs/${job_name}_${ts}.err"

# --- Command to run inside allocation ---
if [[ "$acl_mode" == "true" ]]; then
  # ACL评估模式
  inner_cmd=$(cat <<EOF
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_ACL_test.py \
  --model_path=$model_save_path \
  --acl_mode \
  --acl_root_dir="data/acl-6060/2/acl_6060" \
  --acl_glossary_path="data/acl-6060/2/intermediate_files/terminology_glossary.csv" \
  --acl_test_split="eval" \
  --acl_index_split="dev" \
  --acl_segmentation="gold" \
  --max_eval=1000
EOF
)
else
  # 原有评估模式
  inner_cmd=$(cat <<EOF
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 SONAR_ACL_test.py \
  --model_path=$model_save_path \
  --build_offline_assets \
  --asset_out_dir data \
  --index_type ivfpq \
  --use_ip \
  --nlist 4096 \
  --pq_m 64 \
  --pq_bits 8 \
  --nprobe 16 \
  --test_samples_path=$test_samples_path \
  --max_eval=1000
EOF
)
fi

# Note:
# - srun here requests a fresh allocation and runs non-interactively.
# - Adjust --time as your policy allows; add --qos if your cluster uses it.

echo "Submitting with srun… logs: $out_log / $err_log"
echo "Mode: $([ "$acl_mode" == "true" ] && echo "ACL Evaluation" || echo "Standard Evaluation")"

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

# 显示使用示例
echo ""
echo "=== Usage Examples ==="
echo "Standard evaluation:"
echo "  bash SONAR_ACL_test.sh 2 term true"
echo "ACL evaluation:"
echo "  bash SONAR_ACL_test.sh 2 term true data/samples/xl/term_level_chunks_500000_1000000.json true"
echo ""

exit $exit_code