import json
import os
import re

# === 参数配置 ===
input_path = "data/final_named_entities_info.json"
output_path = "data/filtered_named_entities_multilang.json"

target_langs = {"zh", "de", "es"}
filter_qid_only = True       # 是否过滤 Q123456 类 label

# === 工具函数 ===
def is_qid_like(term):
    return re.fullmatch(r"Q\d{3,}", term)

def decode_unicode_placeholders(term):
    # 匹配 _UXXXX_、_UXXXX、UXXXX_、UXXXX 四种情况
    term = re.sub(r"(?:^|_)U([0-9A-Fa-f]{4})(?:_|$)?", "_", term)
    # 将多个连续的 _ 合并为一个
    term = re.sub(r"_+", "_", term)
    # 去除首尾下划线
    term = term.strip("_")
    return term

# === 主逻辑 ===
with open(input_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

filtered = {}
stats = {"total": 0, "qid_filtered": 0, "lang_missing": 0, "kept": 0}

for term, info in all_data.items():
    stats["total"] += 1

    # 过滤 QID-only 条目
    if filter_qid_only and is_qid_like(term):
        stats["qid_filtered"] += 1
        continue

    # 解码 unicode 占位符
    term_decoded = decode_unicode_placeholders(term)

    # 去掉 Q 编号后缀
    term_clean = re.sub(r"_Q\d{3,}$", "", term_decoded)

    # 去除所有非 ASR 符号：保留字母、数字、下划线、空格
    term_clean = re.sub(r"[^\w\s]", "", term_clean)

    # 跳过非 ASCII 编码
    if not term_clean.isascii():
        continue

    # 清洗后为空也跳过
    if not term_clean.strip():
        continue

    # 检查是否有目标语言 label
    labels = info.get("labels", {})
    label_langs = set(labels.keys())
    langs_present = label_langs & target_langs
    # 只保留目标语言的 labels
    info["labels"] = {lang: text for lang, text in labels.items() if lang in target_langs}
    if len(langs_present) == 0:
        stats["lang_missing"] += 1
        continue

    # 保留清洗后的 term 和原始信息
    filtered[term_clean] = info
    stats["kept"] += 1

# === 保存结果 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

# === 打印统计 ===
print("\n📊 筛选完成：")
for k, v in stats.items():
    print(f"{k}: {v}")