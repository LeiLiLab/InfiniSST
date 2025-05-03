import json
import random
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset


def load_gigaspeech_and_filter(input_path, output_path, max_samples=100):
    with open(input_path, 'r') as f:
        raw = json.load(f)
        data = raw.get("audios")
    print(f"Total documents: {len(data)}")

    ner = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple", device=0)

    def normalize(text):
        return text.replace("<COMMA>", ",").replace("<PERIOD>", ".").replace("<QUESTIONMARK>", "?").title()

    all_items = []
    for doc in tqdm(data, desc="Filtering samples"):
        segments = doc.get("segments", [])
        for item in segments:
            text = item.get("text_tn", "").strip()
            begin_time = item.get("begin_time", 0)
            end_time = item.get("end_time", 0)
            duration = end_time - begin_time

            if text and 2 <= duration <= 5:
                all_items.append({
                    "sid": item.get('sid', ''),
                    "text": normalize(text),
                    "begin_time": begin_time,
                    "end_time": end_time
                })

    all_items = random.sample(all_items, max_samples)

    dataset = Dataset.from_list(all_items)

    def extract_entities(batch):
        results = ner(batch["text"])
        batch_terms = []
        for r in results:
            terms = list({ent['word'] for ent in r})
            batch_terms.append(terms)
        return {"ground_truth_terms": batch_terms}

    dataset = dataset.map(extract_entities, batched=True, batch_size=32)
    samples = dataset.to_list()

    samples = [s for s in samples if s.get("ground_truth_terms")]

    print(f"Found {len(samples)} valid samples.")

    selected = random.sample(samples, min(max_samples, len(samples)))

    with open(output_path, 'w') as f_out:
        json.dump(selected, f_out, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} samples to {output_path}")


# 调用
load_gigaspeech_and_filter(
    input_path="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/GigaSpeech.json",
    output_path="/home/jiaxuanluo/InfiniSST/retriever/data/gigaspeech_test_samples.json",
    max_samples=1000
)