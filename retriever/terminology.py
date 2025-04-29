import wikipedia
from opencc import OpenCC
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
import wikipedia
from opencc import OpenCC
import random
import os
from rdflib import Graph, URIRef


# ---- Config ----
TARGET_LANGS = ["zh", "de", "es","en"]
MAX_WORKERS = 4
SAVE_INTERVAL = 1000
RETRY_LIMIT = 3
cc = OpenCC('t2s')  # 繁转简

graph = Graph()
print("📥 Loading RDF dump...")
graph.parse("wikidata-20240401-truthy.nt.bz2", format="nt")
print("✅ RDF loaded.")

def get_wikidata_entry(qid, target_langs):
    translations = {}
    description = ""
    en_label = None
    try:
        base = "http://www.wikidata.org/entity/"
        label_prefix = "http://www.w3.org/2000/01/rdf-schema#label"
        desc_prefix = "http://schema.org/description"

        # You should load a pre-parsed RDF graph using rdflib or other method
        # This function assumes a `graph` object exists globally, populated from a .ttl or .nt dump

        for _, _, label in graph.triples((URIRef(base + qid), URIRef(label_prefix), None)):
            if label.language == "en":
                en_label = str(label)
            if label.language in target_langs:
                translations[label.language] = str(label)

        for _, _, desc in graph.triples((URIRef(base + qid), URIRef(desc_prefix), None)):
            if desc.language == "en":
                description = str(desc)

        if en_label:
            return {
                "term": en_label,
                "translations": translations if translations else None,
                "summary": description
            }
        else:
            return None
    except Exception as e:
        print(f"❌ Error processing {qid}: {e}")
        return None

def load_titles(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_results(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def build_translation_db(titles, output_path):
    db = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(get_wikidata_entry, t, TARGET_LANGS): t for t in titles}
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                if result:
                    db.append(result)
                else:
                    print(f"⚠️ Skipped term (no translation or summary): {futures[future]}")
                if (i + 1) % SAVE_INTERVAL == 0:
                    print(f"✅ Progress: {i + 1} terms")
                    save_results(db, output_path)
            except Exception as e:
                print(f"❌ Error processing {futures[future]}: {e}")
                sleep(1)
    save_results(db, output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to terms_x.txt')
    parser.add_argument('--output', required=True, help='Path to save results_x.json')
    args = parser.parse_args()

    print(f"📄 Loading terms from {args.input}")
    terms = load_titles(args.input)
    print(f"🔍 Processing {len(terms)} terms")

    build_translation_db(terms, args.output)

    print(f"🎉 Done! Saved to {args.output}")

if __name__ == "__main__":
    main()
