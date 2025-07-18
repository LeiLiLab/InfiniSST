# extract_ner_cache.py
import json
import spacy
from datasets import load_dataset
from tqdm import tqdm
from datasets import Audio
import sys

nlp = spacy.load("en_core_web_trf")


def extract_named_entities(name, split, sample_limit):
    gs = load_dataset("speechcolab/gigaspeech", name=name, trust_remote_code=True)
    gs = gs.cast_column("audio", Audio(decode=False))
    if sample_limit is not None:
        train_set = gs[split].select(range(sample_limit))
    else:
        train_set = gs[split]
    print(f"[INFO] Loaded {len(train_set)} samples from split '{split}' of '{name}'")

    texts = (sample["text"] for sample in train_set)
    named_entities = []
    for doc in tqdm(nlp.pipe(texts, batch_size=32), ncols=100, dynamic_ncols=True, mininterval=1.0):
        ents = set(ent.text.lower() for ent in doc.ents)
        named_entities.append(ents)

    output_path = f"data/named_entities_{name}_{split}_{sample_limit}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([list(x) for x in named_entities], f, indent=2, ensure_ascii=False)

    print(f"✅ NER saved to {output_path}")


if __name__ == "__main__":
    #extract_named_entities(name="m", split="train", sample_limit=None)
    extract_named_entities(name="l", split="train", sample_limit=None)
    # extract_named_entities(name="m", split="validation", sample_limit=None)
    # extract_named_entities(name="m", split="test", sample_limit=None)

    #extract_named_entities(name="dev",split="validation", sample_limit=None)