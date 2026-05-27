#!/usr/bin/env python3
"""
Test Gemini API connection and JSON response.
"""

import os
import json
import re

# Check API key
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set")
    exit(1)

print("Testing Gemini API...")

# Try new API first
try:
    from google import genai
    USE_NEW_API = True
    print("Using new google.genai package")
except ImportError:
    import google.generativeai as genai
    USE_NEW_API = False
    print("Using legacy google.generativeai package")

MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash")

# Initialize client
if USE_NEW_API:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    print(f"Initialized new API client with model: {MODEL_NAME}")
else:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    client = genai.GenerativeModel(MODEL_NAME)
    print(f"Initialized legacy API client with model: {MODEL_NAME}")

# Test prompt
test_prompt = """You are a technical term extraction assistant.

Extract technical terms from this text: "BERT is a transformer-based model for NLP pre-training. It uses attention mechanisms."

Return a JSON array of objects with this format:
[
  {
    "term": "BERT",
    "full_form": "Bidirectional Encoder Representations from Transformers",
    "description": "A transformer-based model for NLP pre-training",
    "is_acronym": true
  }
]

Only return the JSON array, no additional text.
"""

print("\nSending test request...")

try:
    if USE_NEW_API:
        print("\nDEBUG: Trying new API...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=test_prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
        )
        
        print(f"DEBUG: Response type: {type(response)}")
        print(f"DEBUG: Response attributes: {dir(response)}")
        
        # Try different ways to access response text
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            response_text = response.candidates[0].content.parts[0].text
        else:
            print("DEBUG: Response object:", response)
            response_text = str(response)
    else:
        print("\nDEBUG: Using legacy API...")
        response = client.generate_content(
            test_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
        )
        response_text = response.text
    
    print("\nRaw response:")
    print("-" * 80)
    print(response_text)
    print("-" * 80)
    
    # Try to parse JSON
    # Remove markdown code blocks if present
    cleaned = re.sub(r'```json\s*', '', response_text)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Try to find JSON array
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
        print("\nExtracted JSON:")
        print(cleaned)
    
    terms = json.loads(cleaned)
    
    print(f"\n✓ Successfully parsed {len(terms)} terms!")
    for term in terms:
        print(f"  - {term.get('term', 'N/A')}: {term.get('description', 'N/A')[:60]}...")
    
except json.JSONDecodeError as e:
    print(f"\n✗ JSON parsing failed: {e}")
    print(f"Cleaned text was: {cleaned[:500]}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()























