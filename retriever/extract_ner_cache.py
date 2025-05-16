# extract_ner_cache.py
import json
import spacy
from datasets import load_dataset
from tqdm import tqdm

nlp = spacy.load("en_core_web_trf")


def extract_named_entities(name, split, sample_limit):
    gs = load_dataset("speechcolab/gigaspeech", name=name, trust_remote_code=True)
    if sample_limit is not None:
        train_set = gs[split].select(range(sample_limit))
    else:
        train_set = gs[split]

    named_entities = [set(ent.text.lower() for ent in nlp(sample["text"]).ents) for sample in tqdm(train_set)]

    output_path = f"data/named_entities_{name}_{split}_{sample_limit}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([list(x) for x in named_entities], f, indent=2, ensure_ascii=False)

    print(f"✅ NER saved to {output_path}")


if __name__ == "__main__":
    extract_named_entities(name="m", split="train", sample_limit=None)
    extract_named_entities(name="m", split="validation", sample_limit=None)
    extract_named_entities(name="m", split="test", sample_limit=None)

    #extract_named_entities(name="dev",split="validation", sample_limit=None)