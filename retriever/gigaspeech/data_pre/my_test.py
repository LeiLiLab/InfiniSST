import re

raw_terms = [
    "Acronym expansions", "Acronyms", "Action-level Probability Calibration", 
    "Adam optimizer", "ADR", "Adversarial QA", "AL", "AmbigQA", 
    # ... (你的完整列表) ...
    "KinyaBERT", "SMCalFlow", "RoBERTa", "Precision", "Recall", "F1 score"
]

def clean_paper_terms(term_list):
    cleaned_terms = set()
    
    # 1. 定义领域停用词 (Domain Stopwords)
    # 这些词虽然是术语，但在 ACL 语境下过于泛滥，容易造成 FP
    STOP_TERMS = {
        "precision", "recall", "f1 score", "bleu score", "rouge", "accuracy",
        "correctness", "fluency", "robustness", "error rate", "sota",
        "attention mechanism", "neural networks", "deep learning", "transfer learning",
        "natural language processing", "nlp", "asr", "mt", "qa",
        "cpu", "gpu", "tpu",
        "decoder", "encoder", "embeddings", "word embeddings",
        "acronyms", "algorithms", "models", "data", "datasets"
    }

    for term in term_list:
        term_clean = term.strip()
        term_lower = term_clean.lower()
        words = term_clean.split()
        
        # --- 规则 1: 过滤停用词 ---
        if term_lower in STOP_TERMS:
            continue
            
        # --- 规则 2: 长度控制 (针对 2s Chunk) ---
        # 如果超过 3 个单词，通常太长
        if len(words) > 3:
            # 尝试抢救：提取大写缩写 (e.g., "Masked Language Model BERT" -> BERT)
            # 或者直接丢弃
            continue
            
        # --- 规则 3: 纯小写词的严格审查 ---
        # 如果一个词全是小写（e.g., "correctness", "fluency"），通常是通用词，价值低
        # 除非它是特定术语 (e.g., "beam search" 勉强算，但也很通用)
        # 建议：优先保留含大写字母的词 (Named Entities)
        has_upper = any(c.isupper() for c in term_clean)
        if not has_upper and len(words) == 1:
            # 单个小写词，极大概率是垃圾 (e.g., "correctness")
            continue
            
        cleaned_terms.add(term_clean)

    return sorted(list(cleaned_terms))

# 模拟运行
# refined_list = clean_paper_terms(raw_terms)
# print(f"清洗后剩余: {len(refined_list)} 个 (原始: {len(raw_terms)} 个)")
# print(refined_list[:20])