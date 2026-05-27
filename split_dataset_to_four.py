import json
import random
import os
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"
base_dir = "/mnt/gemini/data1/jiaxuanluo/"

output_files = {
    "pool_A_lagging": os.path.join(base_dir, "train_m_zh_v4_refined_pool_A_lagging.jsonl"),
    "pool_A_GT": os.path.join(base_dir, "train_m_zh_v4_refined_pool_A_GT.jsonl"),
    "Pool_B_distract": os.path.join(base_dir, "train_m_zh_v4_refined_Pool_B_distract.jsonl"),
    "pool_B_baseline": os.path.join(base_dir, "train_m_zh_v4_refined_pool_B_baseline.jsonl")
}

system_prompt = "You are a professional simultaneous interpreter. Your task is to translate English audio chunks into accurate and fluent Chinese. Use the ‘term_map’ as a reference for terminology if provided."

def generate_term_map_string(terms):
    if not terms:
        return ""
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
    # Create a deep copy to avoid modifying original in case of reuse
    inst = json.loads(json.dumps(instance))
    messages = inst['messages']
    gt_terms_by_chunk = inst.get('gt_terms_by_chunk', [])
    
    if messages and messages[0]['role'] == 'system':
        messages[0]['content'] = system_prompt
    
    new_messages = [messages[0]]
    audio_idx = 0
    
    for m in messages[1:]:
        if m['role'] == 'user' and "<audio>" in m['content']:
            term_map_str = ""
            if mode == "rag":
                if audio_idx < len(gt_terms_by_chunk):
                    term_map_str = generate_term_map_string(gt_terms_by_chunk[audio_idx])
            elif mode == "trap":
                if all_terms_pool:
                    num_trash = random.randint(3, 8)
                    trash_terms = random.sample(all_terms_pool, min(num_trash, len(all_terms_pool)))
                    term_map_str = "term_map:\n" + "\n".join([f"{en}={zh}" for en, zh in trash_terms])
            elif mode == "empty":
                term_map_str = ""
            
            content = "<audio>"
            if term_map_str:
                content += "\n\n" + term_map_str
            new_messages.append({"role": "user", "content": content})
            audio_idx += 1
        else:
            new_messages.append(m)
            
    inst['messages'] = new_messages
    return inst

# Data Collection
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
                        if t['zh'] not in assistant_contents[i]:
                            for j in range(i + 1, len(assistant_contents)):
                                if t['zh'] in assistant_contents[j]:
                                    has_delay = True
                                    break
        if has_gt:
            if has_delay: pool_a_lagging.append(instance)
            else: pool_a_ordinary.append(instance)
        else:
            pool_b.append(instance)

# Sampling
print(f"Stats: Lagging={len(pool_a_lagging)}, Ordinary={len(pool_a_ordinary)}, PoolB={len(pool_b)}")

group_lagging = random.sample(pool_a_lagging, 6000)
group_gt = random.sample(pool_a_ordinary, 14000)
pool_b_sampled = random.sample(pool_b, 10000)
group_distract = pool_b_sampled[:3000]
group_baseline = pool_b_sampled[3000:]

# Writing Files
groups = [
    (group_lagging, "rag", "pool_A_lagging"),
    (group_gt, "rag", "pool_A_GT"),
    (group_distract, "trap", "Pool_B_distract"),
    (group_baseline, "empty", "pool_B_baseline")
]

for data, mode, name in groups:
    path = output_files[name]
    print(f"Writing {name} to {path}...")
    with open(path, 'w', encoding='utf-8') as f_out:
        for inst in tqdm(data, desc=name):
            processed = process_instance(inst, mode, all_terms_pool)
            f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')

print("\nAll 4 files have been created successfully.")










