import json
import random
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"
output_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_30k_approx.jsonl"

system_prompt = "You are a professional simultaneous interpreter. Your task is to translate English audio chunks into accurate and fluent Chinese. Use the ‘term_map’ as a reference for terminology if provided."

def generate_term_map_string(terms):
    if not terms:
        return ""
    # Deduplicate by term
    seen = set()
    unique_terms = []
    for t in terms:
        if t['term'] not in seen:
            unique_terms.append(t)
            seen.add(t['term'])
    
    lines = ["term_map:"]
    for t in unique_terms:
        lines.append(f"{t['term']}={t['zh']}")
    return "\n".join(lines)

def process_instance(instance, mode, all_terms_pool=None):
    messages = instance['messages']
    gt_terms_by_chunk = instance.get('gt_terms_by_chunk', [])
    
    # Update system prompt
    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = system_prompt
    
    # Inject term_maps
    new_messages = [messages[0]]
    audio_idx = 0
    
    for m in messages[1:]:
        if m['role'] == 'user' and "<audio>" in m['content']:
            term_map_str = ""
            if mode == "rag":
                # Real GT terms
                if audio_idx < len(gt_terms_by_chunk):
                    term_map_str = generate_term_map_string(gt_terms_by_chunk[audio_idx])
            elif mode == "trap":
                # Random "trash" terms
                if all_terms_pool:
                    num_trash = random.randint(3, 8)
                    trash_terms = random.sample(all_terms_pool, min(num_trash, len(all_terms_pool)))
                    # trash_terms is list of (en, zh)
                    term_map_str = "term_map:\n" + "\n".join([f"{en}={zh}" for en, zh in trash_terms])
            elif mode == "empty":
                # No term_map
                term_map_str = ""
            
            content = "<audio>"
            if term_map_str:
                content += "\n\n" + term_map_str
            new_messages.append({"role": "user", "content": content})
            audio_idx += 1
        else:
            new_messages.append(m)
            
    instance['messages'] = new_messages
    return instance

# First pass: classify and collect terms
pool_a = [] # Both lagging and ordinary
pool_b = []
all_terms_pool = []

print("Reading and classifying...")
with open(input_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if not line.strip(): continue
        instance = json.loads(line)
        gt_terms = instance.get('gt_terms_by_chunk', [])
        has_gt = any(len(chunk) > 0 for chunk in gt_terms)
        
        if has_gt:
            pool_a.append(instance)
            for chunk in gt_terms:
                for t in chunk:
                    if t.get('term') and t.get('zh'):
                        all_terms_pool.append((t['term'], t['zh']))
        else:
            pool_b.append(instance)

# Sampling
print(f"Pool A: {len(pool_a)}, Pool B: {len(pool_b)}")
# Group 1: All Pool A
group1 = pool_a

# Group 2 & 3: Sample from Pool B
num_trap = 3000
num_empty = 7000
pool_b_sampled = random.sample(pool_b, min(num_trap + num_empty, len(pool_b)))

group2 = pool_b_sampled[:num_trap]
group3 = pool_b_sampled[num_trap:]

print("Writing output...")
with open(output_path, 'w', encoding='utf-8') as f_out:
    # Process Group 1 (RAG)
    for inst in tqdm(group1, desc="Group 1 (RAG)"):
        processed = process_instance(inst, "rag")
        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')
    
    # Process Group 2 (Trap)
    for inst in tqdm(group2, desc="Group 2 (Trap)"):
        processed = process_instance(inst, "trap", all_terms_pool)
        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')
        
    # Process Group 3 (Empty)
    for inst in tqdm(group3, desc="Group 3 (Empty)"):
        processed = process_instance(inst, "empty")
        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')

print(f"Done. Refined dataset saved to: {output_path}")
print(f"Final Count: {len(group1) + len(group2) + len(group3)}")










