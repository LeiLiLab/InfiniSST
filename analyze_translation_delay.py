import json
from tqdm import tqdm

#input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_cleaned.jsonl"
input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"
total_gt_terms = 0
delayed_gt_terms = 0
not_found_at_all = 0
found_on_time = 0
total_delay_chunks = 0
delay_counts = {}

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
        
        # Extract assistant contents
        assistant_contents = [m['content'] for m in messages if m['role'] == 'assistant']
        
        # Match each chunk's gt_terms with the assistant response
        # Note: messages[0] is system, messages[1] is user audio 0, messages[2] is assistant audio 0
        # assistant_contents[i] corresponds to gt_terms_by_chunk[i]
        if len(gt_terms_by_chunk) == 0:
            continue
        assert len(assistant_contents) == len(gt_terms_by_chunk), f"Length mismatch: {len(assistant_contents)} != {len(gt_terms_by_chunk)}"
        num_chunks = min(len(assistant_contents), len(gt_terms_by_chunk))
        
        for i in range(num_chunks):
            terms = gt_terms_by_chunk[i]
            for term_obj in terms:
                zh_term = term_obj.get('zh', '')
                if not zh_term:
                    continue
                
                total_gt_terms += 1
                
                # Check if it's in the current chunk
                if zh_term in assistant_contents[i]:
                    found_on_time += 1
                else:
                    # Check if it's in any subsequent chunk
                    delay = -1
                    for j in range(i + 1, len(assistant_contents)):
                        if zh_term in assistant_contents[j]:
                            delay = j - i
                            break
                    
                    if delay != -1:
                        delayed_gt_terms += 1
                        total_delay_chunks += delay
                        delay_counts[delay] = delay_counts.get(delay, 0) + 1
                    else:
                        not_found_at_all += 1

print("\n--- Analysis Results ---")
print(f"Total GT terms processed: {total_gt_terms}")
print(f"Found on time (chunk i): {found_on_time} ({found_on_time/total_gt_terms*100:.2f}%)")
print(f"Delayed (found in i+1, i+2, ...): {delayed_gt_terms} ({delayed_gt_terms/total_gt_terms*100:.2f}%)")
print(f"Not found in current or subsequent chunks: {not_found_at_all} ({not_found_at_all/total_gt_terms*100:.2f}%)")

if delayed_gt_terms > 0:
    avg_delay = total_delay_chunks / delayed_gt_terms
    print(f"Average delay (in chunks) for delayed terms: {avg_delay:.2f}")
    print("Delay distribution (chunks):")
    for d in sorted(delay_counts.keys()):
        print(f"  +{d} chunk(s): {delay_counts[d]} ({delay_counts[d]/delayed_gt_terms*100:.2f}% of delayed)")

