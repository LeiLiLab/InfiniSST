#!/bin/bash

# Data cleaning script runner for ground_truth_terms
# This script runs the Python cleaning script with appropriate paths

set -e

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOSSARY_PATH="${SCRIPT_DIR}/data/terms/glossary_merged.json"
INPUT_DIR="${SCRIPT_DIR}/data/samples/xl"
OUTPUT_DIR="${SCRIPT_DIR}/data/samples/xl_cleaned"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting ground truth terms cleaning process..."
echo "Glossary: $GLOSSARY_PATH"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo

# Run the Python script
python3 "${SCRIPT_DIR}/clean_ground_truth_terms.py" \
    --glossary "$GLOSSARY_PATH" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Cleaning process completed!"
echo "Cleaned files are saved in: $OUTPUT_DIR"

