#!/usr/bin/env python3
"""List available Gemini models."""

import os
import google.generativeai as genai

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set")
    exit(1)

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print("Available Gemini models:")
print("=" * 80)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"\n✓ {model.name}")
        print(f"  Display name: {model.display_name}")
        print(f"  Description: {model.description[:100]}...")
        print(f"  Supported methods: {', '.join(model.supported_generation_methods)}")























