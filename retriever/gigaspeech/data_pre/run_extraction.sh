#!/bin/bash
# Script to install dependencies and run glossary extraction

set -e

echo "==================================="
echo "ACL Paper Glossary Extraction"
echo "==================================="
echo ""

# Check for API key
if [ -z "$GEMINI_API_KEY" ] && [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: No API key found!"
    echo ""
    echo "Please set one of the following:"
    echo "  export GEMINI_API_KEY='your-gemini-key'      (recommended)"
    echo "  export OPENAI_API_KEY='your-openai-key'"
    echo "  export ANTHROPIC_API_KEY='your-anthropic-key'"
    echo ""
    echo "Optional environment variables:"
    echo "  LLM_PROVIDER=gemini (default), openai, or anthropic"
    echo "  MODEL_NAME=gemini-1.5-flash (for Gemini), gpt-4o-mini (for OpenAI), or claude-3-5-sonnet-20241022 (for Anthropic)"
    echo "  OPENAI_API_BASE=http://localhost:8000/v1 (for local/custom endpoints)"
    echo ""
    exit 1
fi

echo "Installing dependencies..."
# Install legacy google-generativeai package for stability
pip uninstall -y google-genai 2>/dev/null || true
pip install PyPDF2 google-generativeai openai anthropic --quiet

echo ""
echo "Configuration:"
echo "  LLM Provider: ${LLM_PROVIDER:-gemini}"
echo "  Model: ${MODEL_NAME:-auto}"
echo ""

echo "Running glossary extraction..."
python extract_acl_terms_from_paper.py

echo ""
echo "Done! Check extracted_glossary.json in the data_pre directory."























