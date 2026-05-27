import json
import os

def extract_and_merge():
    input_jsonl = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v3_gt_terms_final_with_ner.jsonl"
    base_glossary_path = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_used.json"
    output_path = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data/terms/glossary_used_merged_with_gt_terms.json"

    # 1. Extract terms from JSONL
    new_terms_dict = {}
    print(f"Reading terms from {input_jsonl}...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            gt_chunks = data.get("gt_terms_by_chunk", [])
            for chunk in gt_chunks:
                for item in chunk:
                    term = item.get("term")
                    zh = item.get("zh")
                    if term and zh:
                        key = term.lower()
                        if key not in new_terms_dict:
                            new_terms_dict[key] = {
                                "term": term,
                                "classification_reason": "",
                                "confused": False,
                                "short_description": "",
                                "full_form": "",
                                "is_acronym": False,
                                "target_translations": {"zh": zh},
                                "url": ""
                            }
    
    print(f"Extracted {len(new_terms_dict)} unique terms from JSONL.")

    # 2. Load existing glossary
    if os.path.exists(base_glossary_path):
        print(f"Loading existing glossary from {base_glossary_path}...")
        with open(base_glossary_path, 'r', encoding='utf-8') as f:
            glossary = json.load(f)
    else:
        print(f"Base glossary not found at {base_glossary_path}, starting from scratch.")
        glossary = {}

    # 3. Merge terms
    # If key exists in new_terms_dict but not in glossary, add it.
    # If key exists in both, you might want to decide priority. 
    # Usually, we keep existing ones or update them. 
    # The prompt says "最后跟...合并", implying adding missing ones or updating.
    # I'll update the glossary with new terms, but if it already exists, I'll prioritize the new ones as they are "gt_terms".
    
    merged_count = 0
    new_added_count = 0
    for key, value in new_terms_dict.items():
        if key in glossary:
            # Optional: update zh translation or keep original. 
            # I'll update it to ensure GT terms are present.
            glossary[key]["target_translations"]["zh"] = value["target_translations"]["zh"]
            merged_count += 1
        else:
            glossary[key] = value
            new_added_count += 1

    print(f"Merged {merged_count} existing terms, added {new_added_count} new terms.")
    print(f"Total terms in merged glossary: {len(glossary)}")

    # 4. Save merged glossary
    print(f"Saving merged glossary to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)
    print("Done!")

if __name__ == "__main__":
    extract_and_merge()


















