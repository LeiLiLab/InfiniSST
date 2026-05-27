import os
import json
import re
import time
from typing import Dict, List, Any, Optional

# Initialize Gemini client
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai is required. Install it with: pip install google-genai")

if not os.environ.get("GEMINI_API_KEY"):
    print("Error: GEMINI_API_KEY environment variable not set")
    exit(1)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-flash")
print(f"Using Gemini model: {MODEL_NAME}")

# File paths
BASE_DIR = "/mnt/taurus/data/siqiouyang/datasets/acl6060/dev/text/tagged_terminology"
EN_PATH = os.path.join(BASE_DIR, "ACL.6060.dev.tagged.en-xx.en.txt")
ZH_PATH = os.path.join(BASE_DIR, "ACL.6060.dev.tagged.en-xx.zh.txt")
JA_PATH = os.path.join(BASE_DIR, "ACL.6060.dev.tagged.en-xx.ja.txt")
DE_PATH = os.path.join(BASE_DIR, "ACL.6060.dev.tagged.en-xx.de.txt")

OUTPUT_PATH = "/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/data_pre/glossary_acl6060.json"

def _extract_bracketed_terms(text: str) -> List[str]:
    """Extract terms enclosed in [brackets]."""
    return re.findall(r"\[(.*?)\]", text)

def _call_llm_for_terms(en_line: str, zh_line: str, ja_line: str, de_line: str, candidates: List[str]) -> List[Dict[str, Any]]:
    """
    Call LLM to filter candidates and find translations.
    """
    prompt = f"""You are a technical terminology expert for NLP and machine learning.
I have 4 parallel sentences where potential technical terms are marked with [brackets].

English: "{en_line}"
Chinese: "{zh_line}"
Japanese: "{ja_line}"
German: "{de_line}"

Initial English candidates found in brackets: {json.dumps(candidates, ensure_ascii=False)}

Task:
1. Filter the English candidates. Keep only substantive technical terms, acronyms, or domain-specific concepts.
2. DISCARD common words that are obviously not technical terms (e.g., "[For]", "[And]", "[The]" at the start of a sentence).
3. For each valid English term, find its most accurate translation in the corresponding Chinese, Japanese, and German sentences. Note that the translations might also be marked with [brackets] in those sentences.

Return a JSON array of objects. Each object must have these keys:
- "term": the original English term (preserve casing)
- "zh": Chinese translation
- "ja": Japanese translation
- "de": German translation

Return ONLY a valid JSON array, no markdown, no explanation.
Example:
[
  {{"term": "lexical borrowing", "zh": "词汇借用", "ja": "語彙借用", "de": "lexikalische Entlehnung"}},
  {{"term": "NLP task", "zh": "自然语言处理任务", "ja": "NLPタスク", "de": "NLP-Aufgabe"}}
]
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        
        text = response.text.strip()
        # Basic cleanup if needed (though mime_type should handle it)
        if "```json" in text:
            text = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL).group(1)
        elif "```" in text:
            text = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL).group(1)
            
        return json.loads(text)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return []

def main():
    print("Loading parallel tagged transcripts...")
    def read_lines(p):
        with open(p, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    en_lines = read_lines(EN_PATH)
    zh_lines = read_lines(ZH_PATH)
    ja_lines = read_lines(JA_PATH)
    de_lines = read_lines(DE_PATH)

    assert len(en_lines) == len(zh_lines) == len(ja_lines) == len(de_lines), "Line count mismatch!"
    
    total_lines = len(en_lines)
    print(f"Total lines to process: {total_lines}")

    glossary = {}

    for i in range(total_lines):
        en_l, zh_l, ja_l, de_l = en_lines[i], zh_lines[i], ja_lines[i], de_lines[i]
        
        candidates = _extract_bracketed_terms(en_l)
        if not candidates:
            continue

        print(f"[{i}/{total_lines}] Processing line with candidates: {candidates}")
        
        extracted = _call_llm_for_terms(en_l, zh_l, ja_l, de_l, candidates)
        
        for item in extracted:
            term = item.get("term", "")
            if not term: continue
            
            key = term.lower().strip()
            if key not in glossary:
                glossary[key] = {
                    "term": term,
                    "classification_reason": "",
                    "confused": False,
                    "short_description": "",
                    "full_form": "",
                    "is_acronym": False,
                    "target_translations": {
                        "zh": item.get("zh", ""),
                        "ja": item.get("ja", ""),
                        "de": item.get("de", "")
                    },
                    "url": "",
                    "target_translation_source": "tagged_gold"
                }
                print(f"  + Added: {term} -> zh:{item.get('zh')}, ja:{item.get('ja')}, de:{item.get('de')}")

        # Periodic save
        if i % 20 == 0:
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(glossary, f, ensure_ascii=False, indent=2)
            print(f"--- Checkpoint saved: {len(glossary)} terms ---")
            
        # Sleep slightly to respect rate limits if needed
        time.sleep(0.5)

    # Final save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing complete! Total unique terms: {len(glossary)}")
    print(f"Output saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
