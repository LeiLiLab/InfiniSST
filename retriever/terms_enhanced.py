
terms_path = f"term_list_{self.fallback_mode}.json"

with open(terms_path, "w", encoding="utf-8") as f:
    json.dump(self.term_list, f, ensure_ascii=False, indent=2)