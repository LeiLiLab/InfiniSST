#!/bin/bash
#SBATCH --job-name=build_and_retrieve
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=96GB
#SBATCH --output=logs/build_retrieve_%j.out
#SBATCH --error=logs/build_retrieve_%j.err
#SBATCH --dependency=afterok:36305

###36301

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

# samples prehandle

# step 1 split dataset of from gigaspeech to speed up
sbatch samples_pre_handle.sh


# step 1: 合并多个samples文件，根据数据集的名字来做子文件夹隔离
echo "[INFO] Step 1: Merging smaples into one json"
python3 -c "
import json, glob
name = m
files = sorted(glob.glob(f'data/samples/{name}/term_preprocessed_samples_*.json'))
merged = []
for f in files:
   with open(f, encoding='utf-8') as j:
       merged.extend(json.load(j))
print(f'Merged total {len(merged)} samples')
with open(f'data/{name}_preprocessed_samples_merged.json', 'w', encoding='utf-8') as f:
   json.dump(merged, f, indent=2, ensure_ascii=False)
"


echo "[INFO] Step 2: split the glossary files into 4 chunk files to speed glossary embeddings"
python3 glossary_chunk_handle.py --chunks=4 --text_field=short_description



# glossary embedding
echo "[INFO] Step 1: Merging embeddings into FAISS index"
python3 build_glossary_index.py

echo "[INFO] Step 2: Running retrieval evaluation on 4 GPUs"
python3 new_retrieve.py \
    --input /home/jiaxuanluo/InfiniSST/retriever/final_split_terms/ --mode safe --max_gpu 2