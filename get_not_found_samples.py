import json
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20.jsonl"

samples_never_found = []
samples_translated_early = []

total_gt_terms = 0
found_on_time = 0
delayed_gt_terms = 0
not_found_at_all = 0
early_translations = 0
true_never_found = 0

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
        
        num_chunks = min(len(assistant_contents), len(gt_terms_by_chunk))
        
        for i in range(num_chunks):
            terms = gt_terms_by_chunk[i]
            for term_obj in terms:
                zh_term = term_obj.get('zh', '')
                if not zh_term:
                    continue
                
                total_gt_terms += 1
                
                # Case 1: Found on time
                if zh_term in assistant_contents[i]:
                    found_on_time += 1
                    continue
                
                # Case 2: Found later (delayed)
                found_later = False
                for j in range(i + 1, len(assistant_contents)):
                    if zh_term in assistant_contents[j]:
                        found_later = True
                        break
                
                if found_later:
                    delayed_gt_terms += 1
                    continue
                
                # Case 3: Not in i or later. Check if it was translated early (in 0...i-1)
                found_early = False
                for j in range(0, i):
                    if zh_term in assistant_contents[j]:
                        found_early = True
                        break
                
                sample_info = {
                    "utter_id": instance.get('utter_id'),
                    "chunk_index": i,
                    "term": term_obj['term'],
                    "zh_term": zh_term,
                    "all_assistant_contents": assistant_contents,
                    "gt_terms_at_chunk_i": [t['zh'] for t in terms]
                }
                
                if found_early:
                    early_translations += 1
                    if len(samples_translated_early) < 5:
                        samples_translated_early.append(sample_info)
                else:
                    true_never_found += 1
                    if len(samples_never_found) < 5:
                        samples_never_found.append(sample_info)

print("\n--- Refined Analysis ---")
print(f"Total GT terms: {total_gt_terms}")
print(f"Found on time (i): {found_on_time} ({found_on_time/total_gt_terms*100:.2f}%)")
print(f"Delayed (i+1...): {delayed_gt_terms} ({delayed_gt_terms/total_gt_terms*100:.2f}%)")
print(f"Early translation (0...i-1): {early_translations} ({early_translations/total_gt_terms*100:.2f}%)")
print(f"Never translated/found: {true_never_found} ({true_never_found/total_gt_terms*100:.2f}%)")

print("\n=== Samples: Translated Early (Already appeared in previous chunks) ===")
for s in samples_translated_early:
    print(f"\nID: {s['utter_id']}, Chunk: {s['chunk_index']}")
    print(f"Term: {s['term']} -> {s['zh_term']}")
    print(f"Assistant Contents:")
    for idx, c in enumerate(s['all_assistant_contents']):
        marker = " [TARGET CHUNK]" if idx == s['chunk_index'] else ""
        print(f"  {idx}: {c}{marker}")

print("\n=== Samples: Never Translated (Not found in any assistant output) ===")
for s in samples_never_found:
    print(f"\nID: {s['utter_id']}, Chunk: {s['chunk_index']}")
    print(f"Term: {s['term']} -> {s['zh_term']}")
    print(f"Assistant Contents:")
    for idx, c in enumerate(s['all_assistant_contents']):
        marker = " [TARGET CHUNK]" if idx == s['chunk_index'] else ""
        print(f"  {idx}: {c}{marker}")










