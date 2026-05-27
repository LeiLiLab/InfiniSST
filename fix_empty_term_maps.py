import json
import os
from tqdm import tqdm

files_to_fix = [
    "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_final_merged_shuffled.jsonl",
    "/mnt/gemini/data1/jiaxuanluo/train_m_zh_v4_final_merged_shuffled_sample0.5.jsonl"
]

def fix_term_map_format(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    temp_path = file_path + ".tmp"
    fixed_count = 0
    total_count = 0
    
    print(f"Fixing format in {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(temp_path, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in):
            if not line.strip():
                continue
            
            try:
                instance = json.loads(line)
                messages = instance.get('messages', [])
                modified = False
                
                for msg in messages:
                    if msg['role'] == 'user' and "<audio>" in msg['content']:
                        # Check if term_map exists in content
                        if "term_map:" not in msg['content']:
                            msg['content'] = msg['content'].strip() + "\n\nterm_map:NONE"
                            modified = True
                            fixed_count += 1
                
                f_out.write(json.dumps(instance, ensure_ascii=False) + '\n')
                total_count += 1
                
            except json.JSONDecodeError:
                continue
                
    # Replace original file with fixed one
    os.replace(temp_path, file_path)
    print(f"Done. Fixed {fixed_count} messages in {total_count} instances.")

if __name__ == "__main__":
    for f in files_to_fix:
        fix_term_map_format(f)










