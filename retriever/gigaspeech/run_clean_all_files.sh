#!/bin/bash

# Complete data cleaning script for all term_preprocessed_samples files
# This script processes all specified files and provides comprehensive statistics

set -e

# Define paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOSSARY_PATH="${SCRIPT_DIR}/data/terms/glossary_merged.json"
INPUT_DIR="${SCRIPT_DIR}/data/samples/xl"
OUTPUT_DIR="${SCRIPT_DIR}/data/samples/xl_cleaned"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Ground Truth Terms Cleaning - All Files"
echo "=========================================="
echo "Glossary: $GLOSSARY_PATH"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo

# Check if glossary exists
if [ ! -f "$GLOSSARY_PATH" ]; then
    echo "Error: Glossary file not found at $GLOSSARY_PATH"
    exit 1
fi

# Define all files to process
FILES=(
    "term_preprocessed_samples_0_500000.json"
    "term_preprocessed_samples_500000_1000000.json"
    "term_preprocessed_samples_1000000_1500000.json"
    "term_preprocessed_samples_1500000_2000000.json"
    "term_preprocessed_samples_2000000_2500000.json"
    "term_preprocessed_samples_2500000_3000000.json"
    "term_preprocessed_samples_3000000_3500000.json"
    "term_preprocessed_samples_3500000_4000000.json"
    "term_preprocessed_samples_4000000_4500000.json"
    "term_preprocessed_samples_4500000_5000000.json"
    "term_preprocessed_samples_5000000_5500000.json"
    "term_preprocessed_samples_5500000_6000000.json"
    "term_preprocessed_samples_6000000_6500000.json"
    "term_preprocessed_samples_6500000_7000000.json"
    "term_preprocessed_samples_7000000_7500000.json"
    "term_preprocessed_samples_7500000_8000000.json"
    "term_preprocessed_samples_8000000_end.json"
)

# Initialize counters
TOTAL_FILES=0
PROCESSED_FILES=0
SKIPPED_FILES=0

# Process each file
for FILE in "${FILES[@]}"; do
    INPUT_PATH="${INPUT_DIR}/${FILE}"
    OUTPUT_PATH="${OUTPUT_DIR}/${FILE}"
    
    TOTAL_FILES=$((TOTAL_FILES + 1))
    
    echo "----------------------------------------"
    echo "Processing: $FILE"
    echo "----------------------------------------"
    
    if [ -f "$INPUT_PATH" ]; then
        # Run the Python script for this specific file
        python3 "${SCRIPT_DIR}/clean_ground_truth_terms.py" \
            --glossary "$GLOSSARY_PATH" \
            --input-dir "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --files "$FILE"
        
        PROCESSED_FILES=$((PROCESSED_FILES + 1))
        echo "✓ Successfully processed: $FILE"
    else
        echo "⚠ Warning: File not found, skipping: $INPUT_PATH"
        SKIPPED_FILES=$((SKIPPED_FILES + 1))
    fi
    
    echo
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total files: $TOTAL_FILES"
echo "Processed files: $PROCESSED_FILES"
echo "Skipped files: $SKIPPED_FILES"
echo
echo "All cleaned files are saved in: $OUTPUT_DIR"
echo "Data cleaning process completed!"

