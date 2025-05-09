import sys
import os
import json
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from tqdm import tqdm

ENTITY_CLASS_PATH = "data/all_entity_classes.json"
SAVE_DIR = "data/outputs/"
TTL_FILE = sys.argv[1]  # 传入的拆分后 ttl 文件路径

# 读取 entity class
with open(ENTITY_CLASS_PATH) as f:
    ENTITY_CLASSES = set(URIRef(uri) for uri in json.load(f))

# 解析该 ttl 文件
g = Graph()
print(f"📥 Parsing {TTL_FILE}...")

try:
    g.parse(TTL_FILE, format="ttl")
except Exception as e:
    print(f"[ERROR] Parse failed for {TTL_FILE}: {e}")
    exit(1)
print(f"✅ Parsed {len(g)} triples from {TTL_FILE}")

# 提取 named entity subjects
named_entities = set()
for s, p, o in g.triples((None, RDF.type, None)):
    if o in ENTITY_CLASSES:
        named_entities.add(str(s))

# 提取实体信息
results = {}
for s, p, o in g:
    s_str = str(s)
    if s_str not in named_entities:
        continue
    term = s_str.split("/")[-1]
    if term not in results:
        results[term] = {
            "url": s_str,
            "labels": {},
            "comments": "",
            "altNames": []
        }
    if p == URIRef("http://www.w3.org/2000/01/rdf-schema#label"):
        lang = o.language
        if lang and lang != "en":
            results[term]["labels"][lang] = str(o)
    elif p == URIRef("http://www.w3.org/2000/01/rdf-schema#comment") and o.language == "en":
        results[term]["comments"] = str(o)
    elif p == URIRef("http://schema.org/alternateName") and o.language == "en":
        results[term]["altNames"].append(str(o))

# 保存该文件对应的结果
os.makedirs(SAVE_DIR, exist_ok=True)
save_path = os.path.join(SAVE_DIR, os.path.basename(TTL_FILE) + ".json")
with open(save_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"💾 Saved {len(results)} entities to {save_path}")