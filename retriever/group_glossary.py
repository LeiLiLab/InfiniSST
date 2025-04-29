import json
import re
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import glob

def load_glossary(input_path: str):
    glossary_files = sorted(glob.glob(input_path))
    glossary = []
    for file in glossary_files:
        with open(file, "r", encoding="utf-8") as f:
            glossary.extend(json.load(f))
    return glossary

def normalize_term(term: str) -> str:
    return re.sub(r"\(.*?\)", "", term).strip().lower()

def group_glossary_semantic(input_path: str, output_path: str, model_name="all-MiniLM-L6-v2", threshold=0.8):
    glossary = load_glossary(input_path)

    model = SentenceTransformer(model_name)
    root_to_variants = defaultdict(list)

    root_term_to_embedding = {}

    for item in tqdm(glossary, desc="Encoding terms"):
        raw_term = item["term"]
        root = normalize_term(raw_term)
        if root not in root_term_to_embedding:
            root_term_to_embedding[root] = model.encode(root, convert_to_numpy=True)
        root_to_variants[root].append({
            "term": item["term"],
            "summary": item.get("summary", ""),
            "translations": item.get("translations", None),
            "fallback_used": item.get("fallback_used", False)
        })

    roots = list(root_term_to_embedding.keys())
    embeddings = np.array([root_term_to_embedding[r] for r in roots]).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    _, neighbors = index.search(embeddings, 20)  # top-20 nearest neighbors for each term

    clustered = {}
    assigned = {}
    cluster_id = 0

    for i, root in enumerate(roots):
        if root in assigned:
            continue
        current_cluster = [root]
        assigned[root] = cluster_id
        for j in neighbors[i]:
            if j == i:
                continue
            sim = np.dot(embeddings[i], embeddings[j])
            if sim >= threshold and roots[j] not in assigned:
                current_cluster.append(roots[j])
                assigned[roots[j]] = cluster_id
        merged_key = normalize_term(current_cluster[0])  # 使用第一个 root 作为聚合 key
        clustered[merged_key] = []
        for r in current_cluster:
            clustered[merged_key].extend(root_to_variants[r])
        cluster_id += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_path = "outputs/terms_*.json"
    output_path = "grouped_terms_semantic.json"
    group_glossary_semantic(input_path, output_path)