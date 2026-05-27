import json
import glob
import os

OLD_PROMPT = "You are a professional simultaneous interpreter. Your task is to translate English audio chunks into accurate and fluent Chinese. Use the ‘term_map’ as a reference for terminology if provided. Prioritize the audio: evaluate the terms and incorporate any terms that strictly match the audio context. If no terms match, ignore them completely and translate based on your own understanding."
NEW_PROMPT = "You are a professional simultaneous interpreter. Your task is to translate English audio chunks into accurate and fluent Chinese. Use the ‘term_map’ as a reference for terminology if provided."
def update_file(file_path):
    print(f"Updating {file_path}...")
    temp_path = file_path + ".tmp"
    with open(file_path, 'r', encoding='utf-8') as f_in, open(temp_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "messages" in data:
                    for msg in data["messages"]:
                        if msg["role"] == "system" and msg["content"] == OLD_PROMPT:
                            msg["content"] = NEW_PROMPT
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Error processing line in {file_path}: {e}")
                f_out.write(line)
    
    os.replace(temp_path, file_path)
    print(f"Finished updating {file_path}")

if __name__ == "__main__":
    patterns = [
        "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned*_final.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/dev_s_zh_v4_ner_baseline_aligned*_final.jsonl"
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    if not files:
        print("No files found matching the pattern.")
    else:
        for f in files:
            update_file(f)

