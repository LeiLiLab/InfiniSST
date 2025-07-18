#!/bin/bash
#SBATCH --job-name=merge_samples
#SBATCH --partition=taurus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --output=logs/merge_samples_%j.out
#SBATCH --error=logs/merge_samples_%j.err

name=$1  # ✅ 获取传入参数
text_field=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate infinisst

echo "[INFO] Merging samples for dataset: $name"
python3 -c "
import json, glob
name = '$name'
text_field = '$text_field'
if name:
  if text_field != 'term':
    files = sorted(glob.glob(f'data/samples/{name}/preprocessed_samples_*.json'))
  else:
    files = sorted(glob.glob(f'data/samples/{name}/term_preprocessed_samples_*.json'))
else:
  files = sorted(glob.glob(f'data/samples/preprocessed_samples_*.json'))

merged = []
for f in files:
   with open(f, encoding='utf-8') as j:
       merged.extend(json.load(j))
print(f'Merged total {len(merged)} samples')
if name:
  if text_field != 'term':
    with open(f'data/{name}_preprocessed_samples_merged.json', 'w', encoding='utf-8') as f:
       json.dump(merged, f, indent=2, ensure_ascii=False)
  else:
    with open(f'data/{name}_term_preprocessed_samples_merged.json', 'w', encoding='utf-8') as f:
       json.dump(merged, f, indent=2, ensure_ascii=False)
else:
  with open(f'data/preprocessed_samples_merged.json', 'w', encoding='utf-8') as f:
     json.dump(merged, f, indent=2, ensure_ascii=False)
"