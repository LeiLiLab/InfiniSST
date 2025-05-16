


# clean glossary description
def clean_description(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text or len(text) < 5:
        return ""
    # 只保留前一句（提升语义聚焦）
    first_sentence = text.split('.')[0]
    # 去除与术语重复的内容（如 term, ... 开头）
    cleaned = first_sentence.split(",", 1)[-1].strip() if "," in first_sentence else first_sentence
    # 再次判空
    return cleaned if len(cleaned) >= 5 else ""