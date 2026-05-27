import json
from collections import defaultdict
from tqdm import tqdm

input_file = "/mnt/gemini/data1/jiaxuanluo/term_train_dataset_m1.jsonl"
output_file = "/mnt/gemini/data1/jiaxuanluo/ner_candidates_merged_v4.jsonl"

print(f"Loading terms from {input_file}...")
ner_map = defaultdict(set)

with open(input_file, "r") as f:
    for line in tqdm(f):
        try:
            obj = json.loads(line)
            ner_map[obj["utter_id"]].add(obj["term"])
        except:
            continue

print(f"Writing {len(ner_map)} utterances to {output_file}...")
with open(output_file, "w") as f:
    for utter_id, terms in tqdm(ner_map.items()):
        f.write(json.dumps({
            "utter_id": utter_id,
            "ner_candidates": list(terms)
        }, ensure_ascii=False) + "\n")

print("Done.")








