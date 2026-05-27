import sys

# High-frequency characters that look DIFFERENT in Simplified vs Traditional
# These are the TRADITIONAL versions
TRADITIONAL_ONLY = set(
    "發個這國漢龍門義議觀選設廣區處邊實導層應歸劃準際範統顯現檢驗繼續體變讓屬隨響傳錄標聽職聯號規術視認許訴試該詳語誠談證識讀贊贈趕趨輯輸辦運遊達遠適遲遼遺遙鄧鄭鄰醬鑒鐘鐵鏈鎖銳錄鋼錢鑽鑲閃問閒閣閱闊隊階隔"
)

def check_file(file_path, limit=20000):
    print(f"Checking {file_path}...")
    found_traditional = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                for char in line:
                    if char in TRADITIONAL_ONLY:
                        found_traditional.append((i+1, char))
                        if len(found_traditional) >= 10:
                            break
                if len(found_traditional) >= 10:
                    break
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if found_traditional:
        print(f"  [!] Found Traditional Chinese characters in {file_path}:")
        for line_num, char in found_traditional:
            print(f"      Line {line_num}: Character '{char}'")
    else:
        print(f"  [OK] No common Traditional Chinese characters found in the first {limit} lines.")

if __name__ == "__main__":
    files = [
        "/mnt/gemini/data1/jiaxuanluo/train_s_zh_v4_ner_baseline_aligned_freq_k20.jsonl",
        "/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv"
    ]
    for f in files:
        check_file(f)
