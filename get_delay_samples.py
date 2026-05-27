import json
from tqdm import tqdm

input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20_cleaned.jsonl"

target_delay = 5
samples = []

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
                
                # Check current chunk
                if zh_term in assistant_contents[i]:
                    continue
                
                # Check for specific delay
                found_at_delay = False
                # If target_delay is 5, we look at chunk i + 5
                if i + target_delay < len(assistant_contents):
                    if zh_term in assistant_contents[i + target_delay]:
                        # Also check that it wasn't in i+1...i+4 to be strictly a +5 delay
                        was_earlier = False
                        for prev_delay in range(1, target_delay):
                            if zh_term in assistant_contents[i + prev_delay]:
                                was_earlier = True
                                break
                        
                        if not was_earlier:
                            found_at_delay = True
                
                if found_at_delay:
                    samples.append({
                        "utter_id": instance.get('utter_id'),
                        "chunk_i": i,
                        "target_chunk": i + target_delay,
                        "term": term_obj['term'],
                        "zh_term": zh_term,
                        "all_contents": assistant_contents
                    })
                    if len(samples) >= 3:
                        break
            if len(samples) >= 3:
                break
        if len(samples) >= 3:
            break

print(f"\n=== Samples with Delay +{target_delay} chunks ===")
for s in samples:
    print(f"\nID: {s['utter_id']}, Term: {s['term']} -> {s['zh_term']}")
    print(f"标注在 Chunk {s['chunk_i']}，实际出现在 Chunk {s['target_chunk']}")
    print("Assistant Contents:")
    for idx, content in enumerate(s['all_contents']):
        marker = ""
        if idx == s['chunk_i']: marker = " [ANNOTATED HERE]"
        if idx == s['target_chunk']: marker = " [ACTUALLY APPEARED HERE]"
        print(f"  {idx}: {content}{marker}")










