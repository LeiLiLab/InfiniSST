#!/bin/bash
#
# Merge outputs from multi-GPU processing
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

OUTPUT_BASE="/mnt/gemini/data1/jiaxuanluo/train_s_zh_with_candidates"
FINAL_OUTPUT="${OUTPUT_BASE}.jsonl"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Merging Multi-GPU Outputs${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# Check if all GPU outputs exist
MISSING=0
for i in 0 1 2 3; do
    GPU_FILE="${OUTPUT_BASE}_gpu${i}.jsonl"
    if [ ! -f "$GPU_FILE" ]; then
        echo -e "${YELLOW}Warning: Missing $GPU_FILE${NC}"
        MISSING=$((MISSING + 1))
    else
        LINES=$(wc -l < "$GPU_FILE")
        SIZE=$(du -h "$GPU_FILE" | cut -f1)
        echo -e "GPU $i: ${GREEN}$LINES lines, $SIZE${NC}"
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Warning: $MISSING GPU output file(s) missing!${NC}"
    echo "Proceeding with available files..."
fi

echo ""
echo -e "${GREEN}Merging all GPU outputs...${NC}"

# Merge all GPU outputs (in order: 0, 1, 2, 3)
cat "${OUTPUT_BASE}_gpu0.jsonl" \
    "${OUTPUT_BASE}_gpu1.jsonl" \
    "${OUTPUT_BASE}_gpu2.jsonl" \
    "${OUTPUT_BASE}_gpu3.jsonl" \
    > "$FINAL_OUTPUT" 2>/dev/null || {
    echo -e "${YELLOW}Some files may be missing, merging available files${NC}"
    > "$FINAL_OUTPUT"  # Create empty file
    for i in 0 1 2 3; do
        GPU_FILE="${OUTPUT_BASE}_gpu${i}.jsonl"
        if [ -f "$GPU_FILE" ]; then
            cat "$GPU_FILE" >> "$FINAL_OUTPUT"
        fi
    done
}

TOTAL_LINES=$(wc -l < "$FINAL_OUTPUT")
TOTAL_SIZE=$(du -h "$FINAL_OUTPUT" | cut -f1)

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Merge Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo -e "Output: ${GREEN}$FINAL_OUTPUT${NC}"
echo -e "Total lines: ${GREEN}$TOTAL_LINES${NC}"
echo -e "Total size: ${GREEN}$TOTAL_SIZE${NC}"

# Sample output
echo ""
echo "Sample output (first message):"
head -n 1 "$FINAL_OUTPUT" | jq '.' 2>/dev/null || head -n 1 "$FINAL_OUTPUT"

# Ask about cleanup
echo ""
read -p "Remove individual GPU files? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Cleaning up individual GPU files...${NC}"
    rm -f "${OUTPUT_BASE}_gpu"*.jsonl
    echo "Done!"
else
    echo "Individual GPU files retained."
fi


















