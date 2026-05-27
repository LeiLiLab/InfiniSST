#!/bin/bash

# ==================== 环境配置 ====================
export CONDA_PREFIX="/home/jiaxuanluo/miniconda3/envs/infinisst"
export PATH="$CONDA_PREFIX/bin:/home/jiaxuanluo/miniconda3/condabin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/jiaxuanluo/InfiniSST:${PYTHONPATH:-}"

# mwerSegmenter 配置
export MWERSEGMENTER_ROOT="/home/jiaxuanluo/mwerSegmenter"

# ==================== 评估配置 ====================
REFERENCE_TEXTS="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/txt/ACL.6060.dev.en-xx.zh.txt"
AUDIO_YAML="/mnt/taurus/data/siqiouyang/datasets/acl6060/dev.yaml"
SACREBLEU_TOKENIZER="zh"
LATENCY_UNIT="char"
EVAL_SCRIPT="/home/jiaxuanluo/FBK-fairseq/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py"

#BASE_DIR="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4"
BASE_DIR="/mnt/gemini/data2/jiaxuanluo/infinisst_omni_vllm_rag_sweep_v4_ablation_topk"
# ==================== 运行评估 ====================
echo "[INFO] Starting evaluation for sweep results in ${BASE_DIR}"

# 遍历 curated 和 raw 文件夹
# NOTE: directories are named like curated_m{M}_cs{...}_hs{...}_TIMESTAMP
for d in "${BASE_DIR}"/curated_* "${BASE_DIR}"/raw_*; do
    # 检查文件夹是否存在（防止 glob 匹配失败）
    [ -d "$d" ] || continue
    
    INSTANCE_LOG="$d/instances.log"
    
    # 检查是否有 instances.log 文件
    if [ ! -f "$INSTANCE_LOG" ]; then
        echo "[SKIP] No instances.log in $(basename "$d")"
        continue
    fi
    
    echo "================================================================"
    echo "[INFO] Processing: $(basename "$d")"
    
    # 根据前缀选择 glossary
    if [[ "$(basename "$d")" == curated_* ]]; then
        GLOSSARY="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/extracted_glossary_with_translations.json"
    elif [[ "$(basename "$d")" == raw_* ]]; then
        GLOSSARY="/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"
    else
        echo "[WARN] Unknown prefix for $(basename "$d"), skipping."
        continue
    fi
    
    # 执行评估脚本，并将结果输出到文件夹内的 term_eval.log
    python "${EVAL_SCRIPT}" \
      --simuleval-instances "${INSTANCE_LOG}" \
      --reference "${REFERENCE_TEXTS}" \
      --audio-yaml "${AUDIO_YAML}" \
      --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
      --latency-unit "${LATENCY_UNIT}" \
      --glossary "${GLOSSARY}" \
      --term-mismatch-examples 0 | tee "$d/term_eval.log"
    
    echo "[INFO] Done: $(basename "$d")"
    echo ""
done

echo "[INFO] All evaluations completed."


