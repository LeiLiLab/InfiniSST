import json
import os
import re
from opencc import OpenCC

# === 初始化繁简转换器 ===
cc = OpenCC("t2s")

# === 参数配置 ===
input_path = "data/final_named_entities_info.json"
output_path = "data/filtered_named_entities_multilang.json"

raw_target_langs = {"zh", "zh-hans", "zh-hant", "de", "es"}
final_langs = {"zh", "de", "es"}
filter_qid_only = True  # 是否过滤 Q123456 类 label

# === 工具函数 ===
def is_qid_like(term):
    return re.fullmatch(r"Q\d{3,}", term)


def decode_unicode_placeholders(term):
    def replace(m):
        codepoint = m.group(1).upper()
        if codepoint == "0027":
            return "'"
        else:
            return "_"

    # 替换所有形式的 Unicode 占位符：U0027、_U0027、U0027_、_U0027_
    term = re.sub(r"(?:^|_)U([0-9A-Fa-f]{4})(?:_|$)?", replace, term)

    # 合并多个 _ 并去除首尾
    term = re.sub(r"_+", "_", term)
    term = term.strip("_")

    return term

# === 主逻辑 ===
with open(input_path, "r", encoding="utf-8") as f:
    all_data = json.load(f)

filtered = {}
stats = {"total": 0, "qid_filtered": 0, "lang_missing": 0, "kept": 0}

for term, info in all_data.items():
    stats["total"] += 1

    if filter_qid_only and is_qid_like(term):
        stats["qid_filtered"] += 1
        continue

    term_decoded = decode_unicode_placeholders(term)
    term_clean = re.sub(r"_Q\d{3,}$", "", term_decoded)
    term_clean = re.sub(r"[^\w\s]", "", term_clean)

    if not term_clean.isascii():
        continue
    if len(re.findall(r"[A-Za-z]", term_clean)) < 3:
        continue

    # 筛选和处理目标语言 label
    labels = info.get("labels", {})
    processed_labels = {}
    for lang, text in labels.items():
        if lang in raw_target_langs:
            if lang.startswith("zh"):
                simplified = cc.convert(text)
                processed_labels["zh"] = simplified
            else:
                processed_labels[lang] = text

    if not (set(processed_labels) & final_langs):
        stats["lang_missing"] += 1
        continue

    info["labels"] = {lang: text for lang, text in processed_labels.items() if lang in final_langs}

    filtered[term_clean] = info
    stats["kept"] += 1

# === 保存结果 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

# === 打印统计 ===
print("\n📊 筛选完成：")
for k, v in stats.items():
    print(f"{k}: {v}")