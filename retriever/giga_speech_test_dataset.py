import json
import random
from tqdm import tqdm


def load_gigaspeech_and_filter(input_path, output_path, max_samples=100):
    samples = []

    with open(input_path, 'r') as f:
        raw = json.load(f)  # GigaSpeech.json是整个大JSON，不是jsonl
        data = raw.get("audios")
    print(f"Total documents: {len(data)}")

    for doc in tqdm(data, desc="Filtering samples"):
        segments = doc.get("segments", [])
        for item in segments:
            text = item.get("text_tn", "").strip()
            begin_time = item.get("begin_time", 0)
            end_time = item.get("end_time", 0)
            duration = end_time - begin_time

            if text and 2 <= duration <= 10:
                samples.append({
                    "sid": item.get('sid', ''),
                    "text": text,
                    "begin_time": begin_time,
                    "end_time": end_time
                })

    print(f"Found {len(samples)} valid samples.")

    selected = random.sample(samples, min(max_samples, len(samples)))

    with open(output_path, 'w') as f_out:
        json.dump(selected, f_out, ensure_ascii=False, indent=2)

    print(f"Saved {len(selected)} samples to {output_path}")


# 调用
load_gigaspeech_and_filter(
    input_path="/mnt/taurus/data/siqiouyang/datasets/gigaspeech/GigaSpeech.json",
    output_path="gigaspeech_test_samples.json",
    max_samples=100
)