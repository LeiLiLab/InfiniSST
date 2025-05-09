import os
import json
from rdflib import Graph, URIRef
from rdflib.namespace import RDF
from tqdm import tqdm
import faulthandler
faulthandler.enable()

# ====== 参数设置 ======
BATCH_SAVE = 100_000
SAVE_DIR = "data"
TTL_PATH = ["data/yago-facts.filtered.ttl", "data/yago-beyond-wikipedia.filtered.ttl"]
ENTITY_CLASS_PATH = "data/all_entity_classes.json"
PARTIAL_SAVE_PATH = os.path.join(SAVE_DIR, "entity_info_partial.json")
PROCESSED_PATH = os.path.join(SAVE_DIR, "processed_entities.json")
FINAL_PATH = os.path.join(SAVE_DIR, "final_named_entities_info.json")

# ====== 目录准备 ======
os.makedirs(SAVE_DIR, exist_ok=True)

# ====== 加载 ENTITY_CLASSES ======
with open(ENTITY_CLASS_PATH, "r") as f:
    ENTITY_CLASSES = set(URIRef(uri) for uri in json.load(f))
print(f"✅ Loaded {len(ENTITY_CLASSES)} entity classes.")

# ====== 初始化状态 ======
if os.path.exists(PARTIAL_SAVE_PATH):
    with open(PARTIAL_SAVE_PATH, "r") as f:
        entity_info = json.load(f)
    print(f"🔁 Loaded {len(entity_info)} partial entities.")
else:
    entity_info = {}

if os.path.exists(PROCESSED_PATH):
    with open(PROCESSED_PATH, "r") as f:
        processed_entities = set(json.load(f))
else:
    processed_entities = set()

# ====== 主处理逻辑（逐文件） ======
count = 0
for file in TTL_PATH:
    print(f"📥 Parsing TTL: {file}")
    g = Graph()
    try:
        g.parse(file, format="ttl")
        print(f"✅ Parsed {len(g)} triples.")
    except Exception as e:
        print(f"[ERROR] Failed to parse {file}: {e}")
        continue

    print("🔍 Extracting named entities...")
    named_entities = set()
    for s, p, o in tqdm(g.triples((None, RDF.type, None))):
        if o in ENTITY_CLASSES:
            named_entities.add(str(s))
    print(f"✅ Extracted {len(named_entities)} named entities in {file}")

    print("🚀 Extracting entity details...")
    for s, p, o in tqdm(g):
        try:
            s_str = str(s)
            if s_str in processed_entities or s_str not in named_entities:
                continue

            term = s_str.split("/")[-1]
            if term not in entity_info:
                entity_info[term] = {
                    "url": s_str,
                    "labels": {},
                    "comments": "",
                    "altNames": []
                }

            if p == URIRef("http://www.w3.org/2000/01/rdf-schema#label"):
                lang = o.language
                if lang and lang != "en":
                    entity_info[term]["labels"][lang] = str(o)
            elif p == URIRef("http://www.w3.org/2000/01/rdf-schema#comment"):
                if o.language == "en":
                    entity_info[term]["comments"] = str(o)
            elif p == URIRef("http://schema.org/alternateName"):
                if o.language == "en":
                    entity_info[term]["altNames"].append(str(o))

            processed_entities.add(s_str)
            count += 1

            if count % BATCH_SAVE == 0:
                print(f"💾 Saving checkpoint at {count} entities...")
                with open(PARTIAL_SAVE_PATH, "w") as f:
                    json.dump(entity_info, f, indent=2, ensure_ascii=False)
                with open(PROCESSED_PATH, "w") as f:
                    json.dump(sorted(processed_entities), f)

        except Exception as e:
            print(f"[WARN] Skipped triple ({s}, {p}, {o}) due to error: {e}")
            continue

    # 手动释放 Graph 占用内存
    del g

# ====== 最终保存 ======
print("✅ Saving final result...")
with open(FINAL_PATH, "w") as f:
    json.dump(entity_info, f, indent=2, ensure_ascii=False)
print(f"🎉 Saved full entity metadata to {FINAL_PATH}")

# ====== 清理中间缓存 ======
if os.path.exists(PARTIAL_SAVE_PATH):
    os.remove(PARTIAL_SAVE_PATH)
if os.path.exists(PROCESSED_PATH):
    os.remove(PROCESSED_PATH)
print("🧹 Temp files removed.")