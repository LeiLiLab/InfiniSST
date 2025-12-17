
✅ 伪代码（带详细注释）

# ===============================
# 0. 初始化与配置
# ===============================

# 读取 term-level chunks 源文件（你生成的 xl_cleaned_term_level_chunks_merged.json）
load term_chunks_json

# 读取 glossary（glossary_cleaned.json）
load glossary_dict

# 建立术语到中文翻译的映射
# glossary_dict 的结构是： {term: {"target_translations": {"zh": "..."} } }
translation_map = {
    normalize(term) : glossary_dict[term]["target_translations"]["zh"]
    for term in glossary_dict
}

# 准备两个容器
term_samples = []       # 含有术语的样本
no_term_samples = []    # 不含术语的样本

# ===============================
# 1. 按 segment_id 过滤 term / no-term
# ===============================

for sample in term_chunks_json:

    # 判断是否为 no-term
    if "no_term" in sample["segment_id"]:
        add sample to no_term_samples
    else:
        add sample to term_samples


# ===============================
# 2. 计算需要多少 samples
# ===============================

# 你的目标：总样本 180k，其中 no_term 占 50%
total_target = 180_000
target_no_term = total_target * 0.5         # 90k
target_term = total_target * 0.5            # 90k

# 随机打乱两个池子
shuffle(term_samples)
shuffle(no_term_samples)

# 选取目标数量（如果样本不够则裁剪到最大）
selected_term_samples = term_samples[:target_term]
selected_no_term_samples = no_term_samples[:target_no_term]

# 合并并再次 shuffle
final_samples = shuffle(selected_term_samples + selected_no_term_samples)


# ===============================
# 3. 将 glossary 翻译注入 references
# ===============================

def build_references(sample):
    references = []    # 每个 reference: {"term": "...", "translation": "..."}

    # term chunk 里有可能包含多个术语（通常是 1 个）
    for term in sample["term_chunk_audio_ground_truth_terms"]:

        normalized = normalize(term)

        # 如果 glossary 找不到翻译，跳过
        if normalized not in translation_map:
            continue

        zh = translation_map[normalized]

        references.append({
            "term": term,
            "translation": zh
        })

    return references


# ===============================
# 4. 构造每条训练数据的 JSON 结构
# ===============================

output_dataset = []

for sample in final_samples:

    audio_path = sample["term_chunk_audio"]
    text = sample["term_chunk_text"]       # 作为 assistant response
    references = build_references(sample)  # term + translation

    # user.content 的格式：
    #   "<audio>, references: {...}, {...}"
    user_content = "<audio>"

    if len(references) > 0:
        user_content += ", references: "
        for ref in references:
            user_content += f'{{"term": "{ref["term"]}", "translation": "{ref["translation"]}"}}, '
        user_content = user_content.rstrip(", ")

    # 组装训练条目
    training_item = {
        "messages": [
            {
                "role": "system",
                "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": text
            }
        ],
        "audios": [ audio_path ]
    }

    output_dataset.append(training_item)


# ===============================
# 5. 结果保存
# ===============================

save output_dataset to output.jsonl  # 或 json，根据你的训练框架决定

