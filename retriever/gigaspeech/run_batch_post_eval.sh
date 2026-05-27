#!/bin/bash

# 基础目录设置
BASE_DIR="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_baseline"
SUMMARY_LOG="${BASE_DIR}/all_results_summary.log"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv

# 环境配置 (从原脚本提取)
export CONDA_PREFIX="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv"
export PATH="$CONDA_PREFIX/bin:/mnt/taurus/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# MWER 配置
MWERSEGMENTER_DIR="/home/jiaxuanluo/mwerSegmenter"
export MWERSEGMENTER_ROOT="${MWERSEGMENTER_DIR}"
export PATH="${MWERSEGMENTER_DIR}:${PATH}"

# Glossary 配置
GLOSSARIES=(
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
  "/mnt/taurus/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
)

AUDIO_YAML="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml"

cd "${BASE_DIR}"

# 遍历所有文件夹
for dir in */; do
    # 去除末尾斜杠
    dir=${dir%/}
    
    # 检查是否以数字结尾，如果是则跳过
    if [[ "$dir" =~ [0-9]$ ]]; then
        continue
    fi
    
    # 跳过特定的非结果文件夹
    if [[ "$dir" == "all_results_summary.log" ]]; then
        continue
    fi

    echo "[INFO] Processing directory: $dir"

    # 从文件夹名称解析 MODEL_SHORT, CUR_SEG 和 CUR_LANG
    # 匹配模式: (de|ja|gigaspeech-zh)_seg([0-9\.]+)_
    if [[ "$dir" =~ ^(de|ja|gigaspeech-zh)_seg([0-9\.]+)_$ ]]; then
        MODEL_SHORT="${BASH_REMATCH[1]}"
        CUR_SEG="${BASH_REMATCH[2]}"
        
        if [ "$MODEL_SHORT" == "gigaspeech-zh" ]; then
            CUR_LANG="zh"
        else
            CUR_LANG="$MODEL_SHORT"
        fi
    else
        echo "[WARN] Skipping $dir: pattern mismatch"
        continue
    fi

    # 根据语言设置分词器和单位
    if [ "${CUR_LANG}" == "zh" ]; then
      CUR_TOKENIZER="zh"
      CUR_LATENCY_UNIT="char"
    elif [ "${CUR_LANG}" == "ja" ]; then
      CUR_TOKENIZER="ja-mecab"
      CUR_LATENCY_UNIT="char"
    elif [ "${CUR_LANG}" == "de" ]; then
      CUR_TOKENIZER="13a"
      CUR_LATENCY_UNIT="word"
    fi

    REF_FILE="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.${CUR_LANG}.txt"
    OUTPUT_PATH="${BASE_DIR}/${dir}"

    # 检查是否存在 instances.log
    if [ ! -f "${OUTPUT_PATH}/instances.log" ]; then
        echo "[WARN] instances.log not found in ${OUTPUT_PATH}, skipping..."
        continue
    fi

    # 执行评估逻辑并加锁写入日志
    (
        exec 200>>"${SUMMARY_LOG}"
        flock 200
        
        echo "--------------------------------------------------------" >> "${SUMMARY_LOG}"
        echo "TIMESTAMP: $(date +'%Y-%m-%d %H:%M:%S')" >> "${SUMMARY_LOG}"
        echo "BATCH_PROCESS | MODEL: ${MODEL_SHORT} | SEG: ${CUR_SEG} | LANG: ${CUR_LANG}" >> "${SUMMARY_LOG}"
        echo "OUTPUT_PATH: ${OUTPUT_PATH}" >> "${SUMMARY_LOG}"

        for GLOS in "${GLOSSARIES[@]}"; do
          GLOS_NAME=$(basename "${GLOS}")
          echo ">>> Glossary: ${GLOS_NAME}" >> "${SUMMARY_LOG}"
          
          python /home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py \
            --simuleval-instances "${OUTPUT_PATH}/instances.log" \
            --reference "${REF_FILE}" \
            --audio-yaml "${AUDIO_YAML}" \
            --sacrebleu-tokenizer "${CUR_TOKENIZER}" \
            --latency-unit "${CUR_LATENCY_UNIT}" \
            --glossary "${GLOS}" \
            --term-lang "${CUR_LANG}" \
            --term-mismatch-examples 0 >> "${SUMMARY_LOG}" 2>&1
        done

        echo "--------------------------------------------------------" >> "${SUMMARY_LOG}"
        echo "" >> "${SUMMARY_LOG}"
        
        flock -u 200
    ) 200>&-
done

echo "[SUCCESS] Batch processing finished. Results are in ${SUMMARY_LOG}"








