from rdflib import Graph, URIRef
from rdflib.namespace import RDFS
import json

INPUT_FILE = "data/yago-taxonomy.ttl"
OUTPUT_FILE = "data/all_entity_classes.json"

g = Graph()
print(f"📖 Parsing {INPUT_FILE} ...")
g.parse(INPUT_FILE, format="ttl")

classes = set()

for s, p, o in g.triples((None, RDFS.subClassOf, None)):
    s_str, o_str = str(s), str(o)

    if s_str.startswith("http://yago-knowledge.org/resource/") or s_str.startswith("http://schema.org/"):
        classes.add(s_str)
    if o_str.startswith("http://yago-knowledge.org/resource/") or o_str.startswith("http://schema.org/"):
        classes.add(o_str)

print(f"✅ Extracted {len(classes)} classes.")

# 保存到 json
with open(OUTPUT_FILE, "w") as f:
    json.dump(sorted(classes), f, indent=2)

print(f"📁 Saved to {OUTPUT_FILE}")