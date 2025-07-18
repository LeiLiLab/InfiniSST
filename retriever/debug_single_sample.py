import os
import torch
import numpy as np
import torch.nn.functional as F
from glossary_utils import load_clean_glossary_from_file

TOP_K = 10
SAMPLE_SID = "AUD0000001221_S0000432"  # ✅ 更换为你想排查的 segment_id

def main():
    # 1. 加载 glossary 和 FAISS index
    term_set, alt2main, parsed_glossary = load_clean_glossary_from_file(
        "data/terms/term_set.txt",
        "data/terms/alt2main.json",
        "data/terms/glossary_filtered.json"
    )
    parsed_glossary = list(parsed_glossary.values())
    retriever = Retriever(enable_fusion=True, device="cuda", max_gpus=1)
    retriever.term_list = parsed_glossary
    retriever.load_index("data/retriever.index")

    # 2. 检查 term_list 与 index 是否对齐
    print(f"[DEBUG] FAISS index size: {retriever.index.ntotal}")
    print(f"[DEBUG] term_list size: {len(retriever.term_list)}")
    assert retriever.index.ntotal == len(retriever.term_list)

    # 3. 加载 sample 并找到指定的 sid
    test_samples = load_preprocessed_samples("data/preprocessed_samples_merged.json")
    test_samples = [s for s in test_samples if s.get("ground_truth_term")]
    sample = next(s for s in test_samples if s["segment_id"] == SAMPLE_SID)

    print(f"[DEBUG] SID: {sample['segment_id']}")
    print(f"[DEBUG] Text: {sample['text']}")
    print(f"[DEBUG] GT Terms: {sample['ground_truth_term']}")

    # 4. 构造 audio embedding
    audio_tensor = torch.tensor(sample["audio_tensor"]).unsqueeze(0).to("cuda")
    with torch.no_grad():
        padded_audio = F.pad(audio_tensor, (0, 160000 - audio_tensor.shape[-1]))  # padding 到 10 秒
        emb = retriever.model.get_audio_embedding_from_data(x=padded_audio, use_tensor=True)
        emb = F.normalize(emb, dim=-1)

    print(f"[DEBUG] Audio emb shape: {emb.shape}")
    print(f"[DEBUG] FAISS index dim: {retriever.index.d}")
    assert emb.shape[-1] == retriever.index.d

    # 5. 查询 FAISS
    D, I = retriever.index.search(emb.cpu().numpy(), TOP_K)
    retrieved_terms = [retriever.term_list[i] for i in I[0]]
    retrieved_texts = [t["short_description"] for t in retrieved_terms]

    print(f"[DEBUG] Top-{TOP_K} Retrieved:")
    for i, text in enumerate(retrieved_texts):
        print(f"  {i+1:02d}: {text}")

    # 6. Recall 计算
    matched = 0
    gt_lower = [gt.lower() for gt in sample["ground_truth_term"]]
    for gt in gt_lower:
        if gt in [r.lower() for r in retrieved_texts]:
            matched += 1

    recall = matched / len(gt_lower)
    print(f"\n[RESULT] Recall@{TOP_K} = {recall:.2%}  ({matched}/{len(gt_lower)})")


if __name__ == "__main__":
    import json
    # 加载 term list（最终进入 retriever.term_list 的 short_description）
    with open("data/alignment_terms.json") as f:
        indexed_descs = {item["short_description"].strip().lower() for item in json.load(f)}

    # 收集训练数据中所有用过的 short_description
    from new_giga_speech import load_preprocessed_samples

    samples = load_preprocessed_samples("data/preprocessed_samples_merged.json", with_tensor=False)

    train_terms = set()
    for s in samples:
        if s.get("ground_truth_term"):
            train_terms.update(t.strip().lower() for t in s["ground_truth_term"])

    # 比较
    missed = train_terms - indexed_descs
    print(f"[CHECK] Total GT terms: {len(train_terms)}")
    print(f"[CHECK] Missing from index: {len(missed)}")
    print("[EXAMPLES]", list(missed)[:10])

    #main()