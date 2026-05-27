import json
from tqdm import tqdm
from collections import defaultdict

input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"

pool_a_lagging = []
pool_a_ordinary = []
pool_b = []

all_terms = [] # For generating trash maps later

with open(input_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if not line.strip():
            continue
        try:
            instance = json.loads(line)
        except json.JSONDecodeError:
            continue
            
        messages = instance.get('messages', [])
        gt_terms_by_chunk = instance.get('gt_terms_by_chunk', [])
        assistant_contents = [m['content'] for m in messages if m['role'] == 'assistant']
        
        has_gt = False
        has_delay = False
        
        num_chunks = min(len(assistant_contents), len(gt_terms_by_chunk))
        
        for i in range(num_chunks):
            terms = gt_terms_by_chunk[i]
            if terms:
                has_gt = True
                for term_obj in terms:
                    zh_term = term_obj.get('zh', '')
                    en_term = term_obj.get('term', '')
                    if not zh_term: continue
                    all_terms.append((en_term, zh_term))
                    
                    # Check if delayed
                    if zh_term not in assistant_contents[i]:
                        # Check subsequent chunks
                        found_later = False
                        for j in range(i + 1, len(assistant_contents)):
                            if zh_term in assistant_contents[j]:
                                found_later = True
                                break
                        if found_later:
                            has_delay = True
        
        if has_gt:
            if has_delay:
                pool_a_lagging.append(instance)
            else:
                pool_a_ordinary.append(instance)
        else:
            pool_b.append(instance)

print(f"\n--- Statistics ---")
print(f"Pool A (Lagging): {len(pool_a_lagging)}")
print(f"Pool A (Ordinary): {len(pool_a_ordinary)}")
print(f"Pool B (No-GT): {len(pool_b)}")
print(f"Total Unique Terms Collected: {len(set(all_terms))}")










