import glob
import json
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def load_clean_glossary_from_file(path="data/cleaned_glossary.json"):
    with open(path, "r", encoding="utf-8") as f:
        glossary = json.load(f)
    return glossary


def load_glossary_by_dir(input_file):
    glossary_files = sorted(glob.glob(input_file + "*.json"))
    glossary = []
    for file in glossary_files:
        with open(file, "r", encoding="utf-8") as f:
            glossary.extend(json.load(f))
    return glossary

def load_glossary(glossary_path: str):
    with open(glossary_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_and_clean_glossary(input_file, max_terms=None, max_ngram=5):
    """
    Load glossary from file or directory, filter out unwanted terms, and return a cleaned list of dicts with 'term' and 'summary'.
    """
    if input_file.endswith(".json"):
        glossary = load_glossary(input_file)
    else:
        glossary = load_glossary_by_dir(input_file)


    punct_set = set(string.punctuation)
    glossary = [item for item in glossary if is_valid_term(item["term"], punct_set)]
    glossary = glossary[:max_terms] if max_terms else glossary

    glossary = [
        item for item in glossary
        if is_valid_summary(item.get("short_description", "")) and is_valid_ngram(item["term"], max_ngram)
    ]

    import os
    os.makedirs("data", exist_ok=True)
    with open("data/cleaned_glossary.json", "w", encoding="utf-8") as f:
        json.dump(glossary, f, ensure_ascii=False, indent=2)

    return glossary


def is_valid_ngram(term, max_n):
    return len(term.split()) <= max_n



def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def is_stopword_like(term):
    return term in stop_words or len(term) <= 3


def is_valid_term(term,punct_set):
    term = term.strip().lower()
    if term.startswith("category:"):
        return False
    if ":" in term:
        blacklist_prefixes = {
            "talk", "user", "user talk", "wikipedia", "wikipedia talk",
            "file", "file talk", "template", "template talk",
            "category", "category talk", "portal", "module", "help",
            "book", "draft", "timedtext", "mediawiki"
        }
        prefix = term.split(":", 1)[0]
        if prefix in blacklist_prefixes:
            return False
    if not is_ascii(term):
        return False
    if all(w in punct_set for w in term.split()):
        return False
    if is_stopword_like(term):
        return False
    return True


# Additional filters
def is_valid_summary(summary):
    if not summary or len(summary.strip()) < 5:
        return False
    if summary.strip().lower() == "wikimedia disambiguation page":
        return False
    return True

if __name__ == "__main__":
    # Run the cleaning function
    cleaned = load_and_clean_glossary("final_split_terms/")
    print("Cleaned Glossary Done!")