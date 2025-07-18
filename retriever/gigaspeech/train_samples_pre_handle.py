import os
import re
import json
import hashlib
from tqdm import tqdm
import soundfile as sf
import nltk
from nltk.corpus import stopwords

from glossary_utils import load_clean_glossary_from_file

# 确保下载停用词
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('english'))


def should_filter_term(term):
    """
    判断是否应该过滤掉某个术语
    过滤规则：
    1. 停用词
    2. 纯数字
    3. 常见的时间词汇
    4. 单个字符
    5. 常见的序数词
    """
    # 转换为小写进行检查
    term_lower = term.lower().strip()
    
    # 过滤单个字符
    if len(term_lower) <= 1:
        return True
    
    # 过滤停用词
    if term_lower in stop_words:
        return True
    
    # 过滤纯数字（包括带逗号的数字）
    if re.match(r'^[\d,]+$', term_lower):
        return True
    
    # 过滤常见的数字词汇
    number_words = {
        'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty',
        'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion',
        'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'
    }
    if term_lower in number_words:
        return True
    
    # 过滤常见的时间词汇
    time_words = {
        'today', 'yesterday', 'tomorrow', 'morning', 'afternoon', 'evening', 'night',
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
        'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'week', 'month', 'year', 'day', 'hour', 'minute', 'second', 'time'
    }
    if term_lower in time_words:
        return True
    
    # 过滤常见的代词和冠词（补充停用词可能遗漏的）
    common_words = {
        'something', 'someone', 'somewhere', 'anything', 'anyone', 'anywhere', 
        'everything', 'everyone', 'everywhere', 'nothing', 'nobody', 'nowhere'
    }
    if term_lower in common_words:
        return True
    
    return False


def _process_wrapper(args):
    item, named_entities, return_tensor, phrase2desc = args
    return process_item(item, phrase2desc, return_tensor, named_entities)


def normalize(text):
    text = text.replace("<COMMA>", ",").replace("<PERIOD>", ".").replace("<QUESTIONMARK>", "?")
    text = re.sub(r"<[^>]+>", "", text)  # remove all other <...> tags
    return text.lower()


def build_phrase_desc_index(term_set, alt2main, glossary, text_field):
    phrase2desc = {}
    for phrase in term_set:
        if phrase in glossary:
            phrase2desc[phrase] = glossary[phrase][text_field]
        elif phrase in alt2main and alt2main[phrase] in glossary:
            phrase2desc[phrase] = glossary[alt2main[phrase]][text_field]
    return phrase2desc


def extract_ground_truth_terms(text, phrase2desc, named_entities):
    if not named_entities:
        return None

    # Tokenization
    tokens = re.findall(r"\b[\w']+\b", text.lower())
    named_entity_phrases = set(' '.join(re.findall(r"\b[\w']+\b", ne.lower())) for ne in named_entities)
    n = len(tokens)

    matched = []
    for i in range(n):
        for j in range(i + 1, min(i + 6, n + 1)):
            phrase = ' '.join(tokens[i:j])
            if phrase not in named_entity_phrases:
                continue  # 剪枝：只考虑 named entity 范围内的片段
            if phrase in phrase2desc:
                matched.append((phrase, i, j))

    matched.sort(key=lambda x: -(x[2] - x[1]))  # 优先选长的
    selected = []
    occupied = set()
    for phrase, start, end in matched:
        if not any(pos in occupied for pos in range(start, end)):
            desc = phrase2desc.get(phrase)
            if desc:
                selected.append((phrase, desc))
                occupied.update(range(start, end))

    filtered = [desc for phrase, desc in selected if phrase in named_entity_phrases]
    
    # 应用术语过滤，移除常见词、数字等
    if filtered:
        filtered = [term for term in filtered if not should_filter_term(term)]
    
    return filtered if filtered else None


def get_layered_audio_path(segment_id, base_dir="/mnt/data/jiaxuanluo/audio"):
    """根据segment_id生成分层的音频文件路径"""
    # 提取audio_id作为第二层目录（假设格式为 AUD0000000468_S0000084）
    audio_id = segment_id.split('_')[0] if '_' in segment_id else segment_id
    
    # 使用audio_id的前3个字符作为第一层目录
    layer1 = audio_id[:3] if len(audio_id) >= 3 else audio_id
    # 使用完整的audio_id作为第二层目录
    layer2 = audio_id
    
    audio_dir = os.path.join(base_dir, layer1, layer2)
    os.makedirs(audio_dir, exist_ok=True)
    
    audio_file = os.path.join(audio_dir, f"{segment_id}.wav")
    return audio_file


def parse_audio_path(audio_path_str):
    """解析 opus:offset:duration 格式的音频路径"""
    if ':' in audio_path_str:
        parts = audio_path_str.split(':')
        if len(parts) >= 3:
            opus_file = parts[0]
            offset_samples = int(parts[1])
            duration_samples = int(parts[2])
            return opus_file, offset_samples, duration_samples
    
    # 如果不是预期格式，返回原路径
    return audio_path_str, 0, None


def extract_and_save_audio(audio_path_str, segment_id, sample_rate=16000):
    """提取并保存音频片段"""
    try:
        opus_file, offset_samples, duration_samples = parse_audio_path(audio_path_str)
        
        if duration_samples is None:
            print(f"[WARNING] 无法解析音频路径格式: {audio_path_str}")
            return None, None, None
        
        # 检查原始文件是否存在
        if not os.path.exists(opus_file):
            print(f"[SKIP] 原始音频文件不存在: {opus_file}")
            return None, None, None
        
        # 生成保存路径
        save_path = get_layered_audio_path(segment_id)
        
        # 如果文件已存在，直接返回路径和时间信息
        if os.path.exists(save_path):
            begin_time = offset_samples / sample_rate
            end_time = (offset_samples + duration_samples) / sample_rate
            return save_path, begin_time, end_time
        
        # 读取音频片段
        # soundfile以样本为单位进行读取
        audio_data, sr = sf.read(opus_file, start=offset_samples, frames=duration_samples)
        
        # 如果采样率不匹配，可能需要重采样（这里假设已经是正确的采样率）
        if sr != sample_rate:
            print(f"[WARNING] 采样率不匹配: 期望{sample_rate}, 实际{sr}, segment_id: {segment_id}")
        
        # 保存音频文件
        sf.write(save_path, audio_data, sr)
        
        # 计算时间信息（以秒为单位）
        begin_time = offset_samples / sr
        end_time = (offset_samples + duration_samples) / sr
        
        return save_path, begin_time, end_time
        
    except Exception as e:
        print(f"[ERROR] 处理音频文件失败 {segment_id}: {e}")
        return None, None, None


def read_tsv_samples(tsv_path):
    """读取TSV文件，返回样本列表"""
    print(f"[INFO] 从 {tsv_path} 读取数据...")
    
    all_samples = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            parts = line.split("\t")
            
            # 跳过表头行或无效行
            if len(parts) < 5 or parts[0] == "id":
                print(f"[INFO] 跳过表头或无效行: {parts[0] if parts else 'empty'}")
                continue
            
            try:
                # 结构: id	audio	n_frames	speaker	src_text	src_lang
                segment_id, audio_path, n_frames, speaker, src_text = parts[:5]
                src_lang = parts[5] if len(parts) > 5 else "en"
                
                # 只保留需要的字段，text进行全小写处理
                all_samples.append({
                    "segment_id": segment_id,
                    "audio": {"path": audio_path},
                    "text": src_text.lower(),  # 全小写处理
                })
            except Exception as e:
                print(f"[WARNING] 跳过第 {line_idx + 1} 行，解析错误: {e}")
                continue
    
    total_size = len(all_samples)
    print(f"[INFO] 总共读取 {total_size} 个样本")
    return all_samples


def process_item(item, phrase2desc, return_tensor=True, named_entities=None):
    """处理单个样本"""
    speech_text = item["text"]
    # text在读取时已经转换为小写，这里只做标点符号规范化
    speech_text = normalize(speech_text)
    ground_truth_terms = extract_ground_truth_terms(speech_text, phrase2desc, named_entities)

    segment_id = item["segment_id"]
    audio_path_str = item.get("audio", {}).get("path") if isinstance(item.get("audio"), dict) else item.get("audio")
    
    # 处理音频文件
    saved_audio_path, begin_time, end_time = extract_and_save_audio(audio_path_str, segment_id)
    
    if saved_audio_path is None:
        print(f"[SKIP] 跳过音频处理失败的样本: {segment_id}")
        return None
    
    # 提取audio_id（从segment_id中提取，假设格式为 AUD0000000468_S0000084）
    audio_id = segment_id.split('_')[0] if '_' in segment_id else segment_id
    
    # 构建新的样本结构
    processed_item = {
        "segment_id": segment_id,
        "text": speech_text,
        "audio": saved_audio_path,
        "begin_time": begin_time,
        "end_time": end_time, 
        "audio_id": audio_id,
        "ground_truth_term": ground_truth_terms or [],
        "has_target": bool(ground_truth_terms)
    }

    return processed_item


def handle_split_samples(term_set_path, alt2main_path, glossary_path, 
                        tsv_path, ner_json_path, split_id, text_field="term"):
    """处理分割的样本数据"""
    term_set, alt2main, glossary = load_clean_glossary_from_file(term_set_path, alt2main_path, glossary_path)
    print(f"Total terms: {len(term_set)}, total entities: {len(glossary)}")

    # 读取分片TSV文件（已经是分割好的文件）
    all_samples = read_tsv_samples(tsv_path)
    print(f"[INFO] 处理分割 {split_id}: 共 {len(all_samples)} 个样本")
    
    # 加载指定的命名实体文件
    if os.path.exists(ner_json_path):
        print(f"[INFO] Loading NER from {ner_json_path}")
        with open(ner_json_path, "r", encoding="utf-8") as f:
            named_entities_list = json.load(f)
    else:
        raise FileNotFoundError(
            f"[ERROR] Named entity file not found: {ner_json_path}. "
            f"Please ensure the file exists."
        )
    
    # 确保样本数量和命名实体数量一致
    if len(all_samples) != len(named_entities_list):
        print(f"[WARNING] 样本数量 ({len(all_samples)}) 与命名实体数量 ({len(named_entities_list)}) 不一致")
        min_length = min(len(all_samples), len(named_entities_list))
        all_samples = all_samples[:min_length]
        named_entities_list = named_entities_list[:min_length]
        print(f"[INFO] 调整为处理 {min_length} 个样本")

    blacklist_path = "data/terms/black_list.txt"
    blacklist = set()
    if os.path.exists(blacklist_path):
        with open(blacklist_path, "r") as f:
            blacklist = set(line.strip() for line in f)

    phrase2desc = build_phrase_desc_index(term_set, alt2main, glossary, text_field)

    # 保持样本和命名实体的正确对应关系
    results = []
    skipped_count = 0
    
    for i, sample in enumerate(all_samples):
        segment_id = sample["segment_id"]
        if segment_id in blacklist:
            skipped_count += 1
            continue
        
        # 使用正确的索引获取对应的命名实体
        if i < len(named_entities_list):
            named_entities = named_entities_list[i]
        else:
            print(f"[WARNING] No NER data for sample {i}, skipping")
            skipped_count += 1
            continue
            
        # 处理样本
        args = (sample, named_entities, False, phrase2desc)
        result = _process_wrapper(args)
        
        if result is not None:
            # 添加原始索引信息用于调试
            result["original_tsv_index"] = i
            results.append(result)
        else:
            skipped_count += 1
    
    print(f"Processed: {len(results)}, Skipped: {skipped_count}, Total: {len(all_samples)}")

    print(f"Total items: {len(results)}")
    return results


def serialize_for_json(samples):
    """序列化样本为JSON格式"""
    # 现在样本结构已经是最终格式，不需要额外处理
    return samples


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_path", type=str, required=True, help="Path to the input TSV file")
    parser.add_argument("--split_id", type=int, default=0, help="Split ID for processing (0-8)")
    parser.add_argument("--ner_json", type=str, required=True, help="Path to the named entities JSON file")
    parser.add_argument(
        '--text_field', type=str, default="term", choices=["term", "short_description"],
        help="Which field to use as input text (term: comma-split title, short_description: full description)"
    )
    args = parser.parse_args()
    
    term_set_path = "data/terms/term_set.txt"
    alt2main_path = "data/terms/alt2main.json"
    glossary_path = "data/terms/glossary_filtered.json"

    samples = handle_split_samples(
        term_set_path=term_set_path,
        alt2main_path=alt2main_path,
        glossary_path=glossary_path,
        tsv_path=args.tsv_path,
        ner_json_path=args.ner_json,
        split_id=args.split_id,
        text_field=args.text_field
    )

    # TODO 只保留有目标的样本
    json_ready = [s for s in samples if s["has_target"]]
    
    # 输出文件命名
    sample_path = f'{args.text_field}_preprocessed_samples' if args.text_field == 'term' else f'preprocessed_samples'
    prefix = "data/samples/xl"
    os.makedirs(prefix, exist_ok=True)
    

    start_idx = args.split_id * 500000
    end_idx = start_idx + len(samples)
    out_path = f"{prefix}/{sample_path}_{start_idx}_{end_idx}.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2, ensure_ascii=False)

    print(f"✅ {out_path} written successfully with {len(json_ready)} samples.")
