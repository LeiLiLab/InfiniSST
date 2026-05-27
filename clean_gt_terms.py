import json
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Load stop words
nltk_stop_words = set(stopwords.words('english'))

#input_path = "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_rate1.0_k20.jsonl"
input_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_rate1.0_k20.jsonl"
output_path = "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_ner_baseline_aligned_freq_k20_cleaned.jsonl"

total_terms_before = 0
total_terms_after = 0
removed_by_stopword = 0
removed_by_not_in_content = 0

with open(input_path, 'r', encoding='utf-8') as f_in, \
     open(output_path, 'w', encoding='utf-8') as f_out:
    
    for line in tqdm(f_in):
        if not line.strip():
            f_out.write(line)
            continue
            
        try:
            instance = json.loads(line)
        except json.JSONDecodeError:
            f_out.write(line)
            continue
            
        messages = instance.get('messages', [])
        gt_terms_by_chunk = instance.get('gt_terms_by_chunk', [])
        
        # Extract assistant contents
        assistant_contents = [m['content'] for m in messages if m['role'] == 'assistant']
        
        new_gt_terms_by_chunk = []
        
        # Number of chunks should match
        num_chunks = min(len(assistant_contents), len(gt_terms_by_chunk))
        
        for i in range(num_chunks):
            current_chunk_terms = gt_terms_by_chunk[i]
            current_assistant_content = assistant_contents[i]
            
            cleaned_terms = []
            for term_obj in current_chunk_terms:
                term_en = term_obj.get('term', '')
                term_zh = term_obj.get('zh', '')
                
                total_terms_before += 1
                
                # Rule 1: Check if term is a stop word
                if term_en.lower() in nltk_stop_words:
                    removed_by_stopword += 1
                    continue
                
                # Rule 2: Check if zh is in the current or any subsequent assistant content
                found_in_any_remaining = False
                for j in range(i, len(assistant_contents)):
                    if term_zh in assistant_contents[j]:
                        found_in_any_remaining = True
                        break
                
                if not found_in_any_remaining:
                    removed_by_not_in_content += 1
                    continue
                
                # If passed both rules, keep the term
                cleaned_terms.append(term_obj)
                total_terms_after += 1
            
            new_gt_terms_by_chunk.append(cleaned_terms)
        
        # If there are trailing chunks in gt_terms_by_chunk that don't have corresponding assistant content
        # we can just leave them empty or ignore them. Given "one-to-one mapping" mentioned by user.
        # If assistant_contents is longer, we just keep the new_gt_terms_by_chunk as is.
        # Actually, let's make sure the length matches the original gt_terms_by_chunk length if possible.
        if len(gt_terms_by_chunk) > num_chunks:
            # For chunks that have no corresponding assistant content, they technically "don't appear"
            # so we could empty them.
            for i in range(num_chunks, len(gt_terms_by_chunk)):
                new_gt_terms_by_chunk.append([])
        
        instance['gt_terms_by_chunk'] = new_gt_terms_by_chunk
        f_out.write(json.dumps(instance, ensure_ascii=False) + '\n')

print("\n--- Processing Results ---")
print(f"Total terms before: {total_terms_before}")
print(f"Total terms after: {total_terms_after}")
print(f"Removed by stopword: {removed_by_stopword}")
print(f"Removed because zh not in content: {removed_by_not_in_content}")
print(f"Cleaned file saved to: {output_path}")

