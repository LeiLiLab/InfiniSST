# from retriever import Retriever, evaluate
# import glob
# import json
# import os
# import torch
#
#
# def load_glossary(input_file):
#     glossary_files = sorted(glob.glob(input_file + "*.json"))
#     glossary = []
#     for file in glossary_files:
#         with open(file, "r", encoding="utf-8") as f:
#             glossary.extend(json.load(f))
#     return glossary
#
#
# def handle(input_file, mode):
#     glossary = load_glossary(input_file)
#
#     parsed_glossary = []
#     for item in glossary:
#         parsed_glossary.append({
#             "term": item["term"],
#             "summary": item.get("short_description", "")
#         })
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     print(f"\n🔍 Running in {mode} mode on {device}")
#     retriever = Retriever(fallback_mode=mode, device=device)
#     index_file = f"retriever_{mode}.index"
#     terms_file = f"term_list_{mode}.json"
#
#     if os.path.exists(index_file) and os.path.exists(terms_file):
#         print("✅ Loading existing index...")
#         retriever.load_index()
#     else:
#         print("⚙️ Building new index...")
#         retriever.build_index(parsed_glossary)
#         retriever.save_index()
#
#
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', required=True)
#     parser.add_argument('--mode', required=True)
#     args = parser.parse_args()
#
#     handle(input_file=args.input, mode=args.mode)