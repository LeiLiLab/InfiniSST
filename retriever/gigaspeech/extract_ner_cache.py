# extract_ner_cache.py
import argparse
import json
import spacy
import os
from tqdm import tqdm

nlp = spacy.load("en_core_web_trf")


def extract_named_entities(tsv_path):
    # 从本地TSV文件读取数据
    # 格式: id	audio	n_frames	speaker	src_text	src_lang
    # 示例: POD0000000001_S0000008	/mnt/taurus/data/siqiouyang/datasets/gigaspeech/audio/podcast/P0001/POD0000000001.opus:2544000:136320	136320	N/A	DOUGLAS MCGRAY IS GOING TO BE OUR GUIDE YOU WALK THROUGH THE DOOR, YOU SEE THE RED CARPETING, YOU SEE SOMEONE IN A SUIT. THEY MAY BE GREETING YOU.	en
    
    print(f"[INFO] 从 {tsv_path} 读取数据...")
    
    # 读取所有数据
    all_samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:  # 跳过表头（如果有的话）
                parts = line.strip().split("\t")
                if parts[0] == "id":  # 确实是表头
                    continue
                # 如果不是表头，重新处理这一行
                line_idx = -1
            
            try:
                parts = line.strip().split("\t")
                if len(parts) >= 5:  # 确保有足够的列
                    id, audio, n_frames, speaker, src_text = parts[:5]
                    src_lang = parts[5] if len(parts) > 5 else "en"
                    all_samples.append({
                        "id": id,
                        "audio": audio,
                        "n_frames": n_frames,
                        "speaker": speaker,
                        "text": src_text,
                        "src_lang": src_lang
                    })
            except Exception as e:
                print(f"[WARNING] 跳过第 {line_idx + 1} 行，解析错误: {e}")
                continue
    
    total_size = len(all_samples)
    print(f"[INFO] 总共读取 {total_size} 个样本")
    
    # 提取文本并进行命名实体识别
    texts = [sample["text"] for sample in all_samples]
    named_entities = []
    
    print("[INFO] 开始提取命名实体...")
    for doc in tqdm(nlp.pipe(texts, batch_size=32, n_process=4), total=len(texts), ncols=100, dynamic_ncols=True, mininterval=1.0):
        ents = set(ent.text.lower() for ent in doc.ents)
        named_entities.append(ents)
    
    # 确保输出目录存在
    os.makedirs("data", exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(tsv_path))[0]
    output_path = f"data/named_entities_{base_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([list(x) for x in named_entities], f, indent=2, ensure_ascii=False)
    
    print(f"✅ 命名实体已保存到 {output_path}")
    print(f"[INFO] 处理了 {len(all_samples)} 个样本，提取了 {len(named_entities)} 组命名实体")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从本地TSV文件提取命名实体")
    parser.add_argument("--tsv_path", type=str, required=True, help="TSV file path containing samples")
    args = parser.parse_args()
    extract_named_entities(tsv_path=args.tsv_path)