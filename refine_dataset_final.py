import json
import random
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"
output_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_refined_30k_final.jsonl"

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
pool_a_lagging = []
pool_a_ordinary = []
pool_b = []
all_terms_pool = []

print("Reading and classifying...")
with open(input_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f):
        if not line.strip(): continue
        instance = json.loads(line)
        gt_terms_by_chunk = instance.get('gt_terms_by_chunk', [])
        assistant_contents = [m['content'] for m in instance['messages'] if m['role'] == 'assistant']
        
        has_gt = False
        has_delay = False
        
        num_chunks = min(len(assistant_contents), len(gt_terms_by_chunk))
        
        for i in range(num_chunks):
            terms = gt_terms_by_chunk[i]
            if terms:
                has_gt = True
                for t in terms:
                    if t.get('term') and t.get('zh'):
                        all_terms_pool.append((t['term'], t['zh']))
                        # Check delay
                        if t['zh'] not in assistant_contents[i]:
                            # Search in subsequent chunks
                            for j in range(i + 1, len(assistant_contents)):
                                if t['zh'] in assistant_contents[j]:
                                    has_delay = True
                                    break
        
        if has_gt:
            if has_delay:
                pool_a_lagging.append(instance)
            else:
                pool_a_ordinary.append(instance)
        else:
            pool_b.append(instance)

# Sampling logic
print(f"Stats: Lagging={len(pool_a_lagging)}, Ordinary={len(pool_a_ordinary)}, PoolB={len(pool_b)}")

num_lagging_sample = 6000
num_ordinary_sample = 14000
num_trap = 3000
num_empty = 7000

group1_vip = random.sample(pool_a_lagging, min(num_lagging_sample, len(pool_a_lagging)))
group1_ord = random.sample(pool_a_ordinary, min(num_ordinary_sample, len(pool_a_ordinary)))
pool_b_sampled = random.sample(pool_b, min(num_trap + num_empty, len(pool_b)))

group2 = pool_b_sampled[:num_trap]
group3 = pool_b_sampled[num_trap:]

print("Writing output...")
with open(output_path, 'w', encoding='utf-8') as f_out:
    # Process Pool A
    for inst in tqdm(group1_vip + group1_ord, desc="Pool A (RAG)"):
        processed = process_instance(inst, "rag")
        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')
    
    # Process Pool B (Trap)
    for inst in tqdm(group2, desc="Group 2 (Trap)"):
        processed = process_instance(inst, "trap", all_terms_pool)
        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')
        
    # Process Pool B (Empty)
    for inst in tqdm(group3, desc="Group 3 (Empty)"):
        processed = process_instance(inst, "empty")
        f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')

print(f"Done. Refined dataset saved to: {output_path}")
print(f"Final Count: {len(group1_vip) + len(group1_ord) + len(group2) + len(group3)}")










