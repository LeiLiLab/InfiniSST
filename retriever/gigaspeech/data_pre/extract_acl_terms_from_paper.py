#!/usr/bin/env python3
"""
Extract technical glossary terms from ACL papers using LLM API.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Set
import PyPDF2

# Configuration
PAPERS_DIR = Path(__file__).parent / "papers"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_FILE = OUTPUT_DIR / "extracted_glossary.json"

# LLM API Configuration
# Supports: openai, anthropic, gemini
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
API_KEY = (os.environ.get("GEMINI_API_KEY") or 
           os.environ.get("OPENAI_API_KEY") or 
           os.environ.get("ANTHROPIC_API_KEY"))
API_BASE_URL = os.environ.get("OPENAI_API_BASE", None)  # For local LLMs or custom endpoints

# Initialize LLM client
client = None
USE_NEW_API = False  # Global flag for Gemini API version

if LLM_PROVIDER == "gemini":
    try:
        # Use legacy package for stability (new API has issues)
        import google.generativeai as genai
        USE_NEW_API = False
            
        if not os.environ.get("GEMINI_API_KEY"):
            print("Error: GEMINI_API_KEY environment variable not set")
            exit(1)
            
        MODEL_NAME = os.environ.get("MODEL_NAME", "models/gemini-2.5-flash")
        
        # Use legacy API
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        client = genai.GenerativeModel(
            MODEL_NAME,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 2000,
            }
        )
        print(f"Using Gemini model: {MODEL_NAME}")
            
    except ImportError as e:
        print(f"Error: google-generativeai package not installed. Run: pip install google-generativeai")
        exit(1)
elif LLM_PROVIDER == "openai":
    try:
        from openai import OpenAI
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        exit(1)
elif LLM_PROVIDER == "anthropic":
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=API_KEY)
        MODEL_NAME = os.environ.get("MODEL_NAME", "claude-3-5-sonnet-20241022")
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        exit(1)
else:
    print(f"Unsupported LLM provider: {LLM_PROVIDER}")
    print("Set LLM_PROVIDER to 'openai', 'anthropic', or 'gemini'")
    exit(1)

EXTRACTION_PROMPT = """You are extracting a glossary for a technical conference talk.

Given the paper content below, list key technical terms that:
- are domain-specific (NLP / speech / ML)
- are likely to be mentioned in oral presentations
- are substantive technical concepts (not common words)
- include acronyms and their full forms

For each term, provide:
1. The term itself
2. A brief 1-sentence description (optional, can be empty string)
3. Whether it's an acronym with its full form (if applicable)

Return a JSON array of objects. Example format:
[
  {{{{
    "term": "BERT",
    "full_form": "Bidirectional Encoder Representations from Transformers",
    "description": "A transformer-based model for NLP pre-training",
    "is_acronym": true
  }}}},
  {{{{
    "term": "attention mechanism",
    "full_form": "",
    "description": "A neural network component that weighs the importance of different inputs",
    "is_acronym": false
  }}}}
]

Only return the JSON array, no additional text.

Paper content:
{paper_content}
"""


def extract_text_from_pdf(pdf_path: Path, max_pages: int = 20) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = min(len(pdf_reader.pages), max_pages)
            
            text_parts = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            # Clean up text
            full_text = re.sub(r'\s+', ' ', full_text)
            return full_text[:15000]  # Limit to ~15k chars to avoid token limits
            
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""


def extract_terms_with_llm(paper_content: str, paper_name: str) -> List[Dict]:
    """Use LLM to extract technical terms from paper content."""
    if not paper_content.strip():
        print(f"Empty content for {paper_name}, skipping...")
        return []
    
    response_text = ""
    try:
        prompt = EXTRACTION_PROMPT.format(paper_content=paper_content)
        
        print(f"Calling LLM for {paper_name} using {LLM_PROVIDER}...")
        
        if LLM_PROVIDER == "gemini":
            # For Gemini, combine system message with user prompt
            full_prompt = "You are a technical term extraction assistant.\n\n" + prompt
            
            # Use legacy google.generativeai API (more stable)
            response = client.generate_content(full_prompt)
            response_text = response.text.strip()
            
        elif LLM_PROVIDER == "openai":
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a technical term extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            response_text = response.choices[0].message.content.strip()
            
        elif LLM_PROVIDER == "anthropic":
            response = client.messages.create(
                model=MODEL_NAME,
                max_tokens=2000,
                temperature=0.3,
                system="You are a technical term extraction assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.content[0].text.strip()
        
        # Debug: Print first part of response
        print(f"  Response preview: {response_text[:200]}...")
        
        # Try to parse JSON from response
        # Remove markdown code blocks if present
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*', '', response_text)
        
        # Try to find JSON array in response
        # Sometimes LLM adds extra text before/after the JSON
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(0)
        
        terms = json.loads(response_text)
        
        if not isinstance(terms, list):
            print(f"Warning: Response is not a list for {paper_name}")
            return []
            
        print(f"Extracted {len(terms)} terms from {paper_name}")
        return terms
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response for {paper_name}: {e}")
        print(f"Full response was:\n{response_text[:1000]}")
        return []
    except Exception as e:
        print(f"Error calling LLM for {paper_name}: {type(e).__name__}: {e}")
        if response_text:
            print(f"Response was:\n{response_text[:1000]}")
        # Print full traceback for debugging
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return []


def merge_terms(all_terms: List[Dict]) -> Dict[str, Dict]:
    """Merge terms from multiple papers, removing duplicates."""
    merged = {}
    
    for term_obj in all_terms:
        term = term_obj.get("term", "").strip()
        if not term:
            continue
            
        # Normalize term for comparison
        term_key = term.lower()
        
        if term_key not in merged:
            # Convert to glossary format
            merged[term_key] = {
                "term": term,
                "classification_reason": "llm_extracted",
                "confused": False,
                "short_description": term_obj.get("description", ""),
                "full_form": term_obj.get("full_form", ""),
                "is_acronym": term_obj.get("is_acronym", False),
                "target_translations": {},
                "url": ""
            }
    
    # Return with original term as key
    result = {}
    for item in merged.values():
        result[item["term"]] = item
    
    return result


def main():
    """Main function to process all papers and extract glossary."""
    print("Starting glossary extraction from ACL papers...")
    print(f"Papers directory: {PAPERS_DIR}")
    
    if not PAPERS_DIR.exists():
        print(f"Error: Papers directory not found: {PAPERS_DIR}")
        return
    
    # Get all PDF files
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    if not pdf_files:
        print("No PDF files found!")
        return
    
    all_terms = []
    
    # Process each paper
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text from PDF
        paper_content = extract_text_from_pdf(pdf_path)
        
        if not paper_content:
            print(f"No content extracted from {pdf_path.name}")
            continue
        
        print(f"Extracted {len(paper_content)} characters")
        
        # Extract terms using LLM
        terms = extract_terms_with_llm(paper_content, pdf_path.name)
        all_terms.extend(terms)
    
    # Merge and deduplicate terms
    print(f"\n\nTotal terms extracted: {len(all_terms)}")
    glossary = merge_terms(all_terms)
    print(f"Unique terms after merging: {len(glossary)}")
    
    # Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Glossary saved to: {OUTPUT_FILE}")
    print(f"✓ Total unique terms: {len(glossary)}")
    
    # Print sample terms
    print("\nSample terms:")
    for i, (term, info) in enumerate(list(glossary.items())[:10]):
        print(f"  - {term}: {info['short_description'][:80]}")
        if i >= 9:
            break


if __name__ == "__main__":
    main()
