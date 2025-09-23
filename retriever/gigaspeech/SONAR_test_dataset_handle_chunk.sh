#!/bin/bash

# 处理测试数据集的term-level chunks

set -euo pipefail

input_samples="data/samples/test_cleaned/term_preprocessed_samples_test.json"
output_samples="data/samples/test_cleaned/term_level_chunks_test.json"

LOG_FILE="${LOG_FILE:-logs/term_level_single.log}"
mkdir -p logs

if [[ ! -f "$output_samples" ]]; then
  echo "[INFO] Processing single slice term-level chunks..." | tee -a "$LOG_FILE"

  # 1) 写一个临时脚本给 sbatch 提交
  job_script="$(mktemp /tmp/term_level_single.XXXXXX.sh)"
  cat > "$job_script" <<'EOS'
#!/bin/bash
set -euo pipefail
cd /home/jiaxuanluo/InfiniSST/retriever/gigaspeech
. ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst
python3 handle_MFA_term_level_chunks.py \
  --input_json=__INPUT_SAMPLES__ \
  --output_json=__OUTPUT_SAMPLES__ \
  --textgrid_dir=/mnt/data/siqiouyang/datasets/gigaspeech/textgrids \
  --output_audio_dir=/mnt/gemini/data1/jiaxuanluo/term_chunks_cleaned
EOS

  # 2) 把占位符替换成真正的路径（让外层变量在这里展开）
  sed -i \
    -e "s|__INPUT_SAMPLES__|${input_samples}|g" \
    -e "s|__OUTPUT_SAMPLES__|${output_samples}|g" \
    "$job_script"
  chmod +x "$job_script"

  # 3) 提交 sbatch（不使用 --wrap，多行更安全）
  mfa_job=$(sbatch \
    --job-name=term_level_single \
    --partition=taurus \
    --mem=32G \
    --cpus-per-task=4 \
    --ntasks=1 \
    --output=logs/term_level_single_%j.out \
    --error=logs/term_level_single_%j.err \
    "$job_script" | awk '{print $4}')

  echo "[INFO] Submitted SLURM job: ${mfa_job}" | tee -a "$LOG_FILE"
fi