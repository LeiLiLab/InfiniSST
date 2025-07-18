import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--text_field', type=str, default="term", choices=["term", "short_description"],
    help="Which field to use as input text (term: comma-split title, short_description: full description)"
)
parser.add_argument("--chunks", type=int, default=4, help="Number of chunks to split into")
args = parser.parse_args()

with open("data/terms/glossary_filtered.json", encoding="utf-8") as f:
    glossary = json.load(f)

# ✅ 支持 term 或 short_description
texts = [ v[args.text_field] for v in glossary.values() ]
chunk_size = (len(texts) + args.chunks - 1) // args.chunks

os.makedirs(f"data/glossary_chunks/{args.text_field}", exist_ok=True)

for i in range(args.chunks):
    chunk = texts[i * chunk_size: (i + 1) * chunk_size]
    with open(f"data/glossary_chunks/{args.text_field}/text_chunk_{i}.json", "w", encoding="utf-8") as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)

print(f"[INFO] Saved {args.chunks} chunks to data/glossary_chunks/, using {'term' if args.use_term else 'short_description'}")