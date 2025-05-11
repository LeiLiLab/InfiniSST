import json
import os
import re
import argparse
from opencc import OpenCC

# === 初始化繁简转换器 ===
cc = OpenCC("t2s")



# === 工具函数 ===

def remove_invalid_unicode(obj):
    if isinstance(obj, str):
        return obj.encode('utf-8', 'ignore').decode('utf-8', 'ignore')
    elif isinstance(obj, dict):
        return {remove_invalid_unicode(k): remove_invalid_unicode(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [remove_invalid_unicode(i) for i in obj]
    else:
        return obj

# === 主处理函数 ===
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict
term_occurrences = defaultdict(list)


def clean_term(raw_term: str, max_words_length: int = 5):
    """
    清洗一个原始术语字符串。返回 (清洗后的 term, 括号中的内容)，否则返回 None。
    """

    def decode_unicode_placeholders(term: str) -> str:
        def replace(m):
            try:
                return chr(int(m.group(1), 16))
            except:
                return ""

        # 匹配任意形式的 uXXXX（不限制前后字符）
        term = re.sub(r"[Uu]([0-9a-fA-F]{4})", replace, term)
        term = re.sub(r"_+", "_", term)  # 连续下划线压缩
        return term.strip("_ ")

    term = decode_unicode_placeholders(raw_term)

    # 删除尾部 Q编号
    term = re.sub(r"[_\s\(]?Q\d{3,}[\)\s_]*$", "", term)

    # 提取括号中的内容（保留第一个匹配项）
    bracket_match = re.search(r"\(([^)]*)\)", term)
    bracket_content = bracket_match.group(1).strip() if bracket_match else ""
    bracket_content = re.sub(r'[_]+', ' ', bracket_content).strip()

    # 去掉括号及其内容
    term = re.sub(r"\([^)]*\)", "", term)

    # 只保留英文字母、数字、左右括号 (一般是消除歧义wikidata加的), 下划线(YAGO用下划线分割words)
    if not re.fullmatch(r"[A-Za-z0-9'() _.]+", term):
        return None

    #把'.'替换成' '
    term = term.replace(".", " ")
    # 按下划线/空格切词
    words = re.split(r"[_\s]+", term)
    words = [w for w in words if w]
    if len(words) > max_words_length:
        return None

    term = " ".join(words).strip()
    if not term or term[0].isdigit():
        return None

    # 至少包含 3 个英文字母
    if len(re.findall(r"[A-Za-z]", term)) < 3:
        return None

    return term, bracket_content

def process_named_entities(input_path: str, max_words_length: int):
    raw_target_langs = {"zh", "zh-hans", "zh-hant", "de", "es"}
    final_langs = {"zh", "de", "es"}

    with open(input_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    cc = OpenCC("t2s")
    stats = {"total": 0, "qid_filtered": 0, "lang_missing": 0, "term_filtered": 0, "kept": 0}

    term_set = set()
    alt2main = dict()
    glossary = dict()

    for raw_term, info in tqdm(all_data.items(), desc="Processing"):
        stats["total"] += 1

        result = clean_term(raw_term, max_words_length)
        if not result:
            continue
        term_clean,bracket_content = result

        if not term_clean:
            stats["term_filtered"] += 1
            continue

        term_key = term_clean.lower()

        labels = info.get("labels", {})
        processed_labels = {}
        for lang, text in labels.items():
            if lang in raw_target_langs:
                if lang.startswith("zh"):
                    processed_labels["zh"] = cc.convert(text)
                else:
                    processed_labels[lang] = text

        if not (set(processed_labels) & final_langs):
            stats["lang_missing"] += 1
            continue

        base_obj = {
            "url": info.get("url", ""),
            "target_translations": {lang: text for lang, text in processed_labels.items() if lang in final_langs},
            "short_description": f"{term_clean}, ({bracket_content}) {info.get('comments', '')}".strip() if bracket_content
            else f"{term_clean}, {info.get('comments', '')}".strip(),
            "term":term_clean
        }

        term_occurrences[term_key].append(base_obj)

        # repeat element
        if len(term_occurrences[term_key]) > 1:
            continue

        glossary[term_key] = base_obj
        term_set.add(term_key)
        stats["kept"] += 1

        for alt in info.get("altNames", []):
            alt_clean_result = clean_term(alt.strip(), max_words_length)
            if not alt_clean_result:
                stats["term_filtered"] += 1
                continue
            alt_clean = alt_clean_result[0]
            alt_key = alt_clean.lower()
            term_set.add(alt_key)
            alt2main[alt_key] = term_key
            stats["kept"] += 1
    print("\n📊 筛选完成：")

    ambiguous_terms = {k: v for k, v in term_occurrences.items() if len(v) > 1}

    with open("data/terms/ambiguous_terms.json", "w", encoding="utf-8") as f:
        json.dump(remove_invalid_unicode(ambiguous_terms), f, indent=2, ensure_ascii=False)


    for k, v in stats.items():
        print(f"{k}: {v}")



    return term_set, alt2main, glossary



# clean for wiki data source
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

def clean_single_file(filepath, max_words_length):
    local_glossary = dict()
    local_term_occurrences = defaultdict(list)

    with open(filepath, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load {filepath}: {e}")
            return {}, {}

        items = data if isinstance(data, list) else [data]

    for item in items:
        raw_term = item.get('term', '')
        result = clean_term(raw_term, max_words_length=max_words_length)
        if not result:
            continue
        term_clean, bracket_content = result

        term_key = term_clean.lower()
        local_term_occurrences[term_key].append(item)

        if len(local_term_occurrences[term_key]) > 1:
            continue

        item["short_description"] = (
            f"{term_clean}, ({bracket_content}) {item.get('short_description', '')}".strip()
            if bracket_content else f"{term_clean}, {item.get('short_description', '')}".strip()
        )
        item["term"] = term_clean
        local_glossary[term_key] = item

    return local_glossary, local_term_occurrences


def clean_for_wiki_data_json(input_dir="data/final_split_terms", output_path="data/terms/glossary_filtered_from_wiki.json", max_words_length=5, num_threads=32):
    glossary = dict()
    term_occurrences = defaultdict(list)

    json_files = [os.path.join(input_dir, fn) for fn in os.listdir(input_dir) if fn.endswith(".json")]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(clean_single_file, file_path, max_words_length) for file_path in json_files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel cleaning"):
            future_result = future.result()
            if not future_result:
                continue
            file_glossary, file_term_occurrences = future_result

            for term, info in file_glossary.items():
                if term in glossary:
                    term_occurrences[term].append(info)  # duplicate
                    continue
                glossary[term] = info
                term_occurrences[term].extend(file_term_occurrences[term])

    ambiguous_terms = {k: v for k, v in term_occurrences.items() if len(v) > 1}
    os.makedirs("data/terms", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(remove_invalid_unicode(glossary), f, indent=2, ensure_ascii=False)
    with open("data/terms/ambiguous_terms_from_wiki.json", "w", encoding="utf-8") as f:
        json.dump(remove_invalid_unicode(ambiguous_terms), f, indent=2, ensure_ascii=False)

    print(f"\n✅ Wiki terms cleaned (multi-threaded). Kept: {len(glossary)}, Ambiguous: {len(ambiguous_terms)}")




# === 命令行入口 ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and reformat named entity glossary.")
    parser.add_argument("--max_words_length", type=int, default=5)
    parser.add_argument("--input", type=str, default="data/final_named_entities_info.json")
    parser.add_argument("--source", type=str, default="yago")
    args = parser.parse_args()

    if args.source == "wiki":
        print("handle from wiki json")
        clean_for_wiki_data_json()
    else:
        term_set, alt2main, glossary = process_named_entities(
            input_path=args.input,
            max_words_length=args.max_words_length
        )

        # 可选：保存调试信息
        with open("data/terms/alt2main.json", "w", encoding="utf-8") as f:
            json.dump(remove_invalid_unicode(alt2main), f, indent=2, ensure_ascii=False)
        with open("data/terms/glossary_filtered.json", "w", encoding="utf-8") as f:
            json.dump(remove_invalid_unicode(glossary), f, indent=2, ensure_ascii=False)
        with open("data/terms/term_set.txt", "w", encoding="utf-8") as f:
            for term in sorted(term_set):
                f.write(term + "\n")