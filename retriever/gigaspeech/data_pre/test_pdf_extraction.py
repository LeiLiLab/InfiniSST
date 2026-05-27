#!/usr/bin/env python3
"""
Test PDF text extraction without calling LLM API.
"""

import re
from pathlib import Path
import PyPDF2

PAPERS_DIR = Path(__file__).parent / "papers"

def extract_text_from_pdf(pdf_path: Path, max_pages: int = 5) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = min(len(pdf_reader.pages), max_pages)
            
            print(f"  Total pages: {len(pdf_reader.pages)}, extracting first {num_pages} pages")
            
            text_parts = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                text_parts.append(text)
            
            full_text = "\n".join(text_parts)
            # Clean up text
            full_text = re.sub(r'\s+', ' ', full_text)
            return full_text
            
    except Exception as e:
        print(f"  Error reading {pdf_path}: {e}")
        return ""


def main():
    print("Testing PDF extraction...")
    print(f"Papers directory: {PAPERS_DIR}\n")
    
    if not PAPERS_DIR.exists():
        print(f"Error: Papers directory not found: {PAPERS_DIR}")
        return
    
    pdf_files = list(PAPERS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        
        if text:
            print(f"  Extracted {len(text)} characters")
            print(f"  Preview: {text[:200]}...")
        else:
            print(f"  Failed to extract text")
        print()
    
    print("✓ PDF extraction test complete!")


if __name__ == "__main__":
    main()























