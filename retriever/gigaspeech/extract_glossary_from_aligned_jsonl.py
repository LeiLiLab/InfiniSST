#!/usr/bin/env python3
import argparse
import json
import os
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description="Extract a glossary JSON from aligned JSONL (gt_terms_by_chunk).")
    parser.add_argument("--input-jsonl", required=True, help="Aligned JSONL path (contains gt_terms_by_chunk).")
    parser.add_argument("--output-json", required=True, help="Output glossary JSON path.")
    parser.add_argument("--target-lang-code", default="zh", help="Target language code key in gt_terms_by_chunk (zh/ja/de).")
    args = parser.parse_args()

    glossary: "OrderedDict[str, dict]" = OrderedDict()

    def pick_translation(item: dict) -> str:
        # Prefer language-specific field, then generic 'translation', then fallback to 'zh'.
        v = item.get(args.target_lang_code)
        if v:
            return str(v).strip()
        v = item.get("translation")
        if v:
            return str(v).strip()
        if args.target_lang_code == "zh":
            v = item.get("zh")
            if v:
                return str(v).strip()
        return ""

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            gt_terms_by_chunk = obj.get("gt_terms_by_chunk", []) or []
            for chunk in gt_terms_by_chunk:
                if not isinstance(chunk, list):
                    continue
                for item in chunk:
                    if not isinstance(item, dict):
                        continue
                    term = str(item.get("term", "")).strip()
                    if not term:
                        continue
                    translation = pick_translation(item)
                    if not translation:
                        continue

                    key = term.strip()
                    if key not in glossary:
                        glossary[key] = {"translation": translation}

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f_out:
        json.dump(glossary, f_out, ensure_ascii=False, indent=2)

    print(f"[INFO] Wrote glossary with {len(glossary)} terms to {args.output_json}")


if __name__ == "__main__":
    main()









