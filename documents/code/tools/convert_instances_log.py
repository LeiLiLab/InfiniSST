#!/usr/bin/env python3
"""
Convert SimulEval instances.log from Unicode escape format to human-readable format.

Usage:
    python convert_instances_log.py <input_file> [output_file]
    
Examples:
    # Convert with default output name (back.log)
    python convert_instances_log.py /path/to/instances.log
    
    # Convert with custom output name
    python convert_instances_log.py /path/to/instances.log /path/to/readable.log
"""

import sys
import json
import os
from pathlib import Path


def convert_instances_log(input_file: str, output_file: str = None, indent: int = 2) -> None:
    """
    Convert instances.log from Unicode escape to human-readable format.
    
    Args:
        input_file: Path to input instances.log file
        output_file: Path to output file (default: same dir as input, named back.log)
        indent: JSON indentation level (default: 2)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    
    if output_file is None:
        output_file = input_path.parent / "back.log"
    else:
        output_file = Path(output_file)
    
    print(f"Converting: {input_path}")
    print(f"Output to: {output_file}")
    
    converted_lines = 0
    failed_lines = 0
    
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:
            
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                
                # Preserve empty lines
                if not line:
                    fout.write('\n')
                    continue
                
                try:
                    # Parse JSON and re-dump with ensure_ascii=False for readable Chinese
                    obj = json.loads(line)
                    fout.write(json.dumps(obj, ensure_ascii=False, indent=indent) + '\n')
                    converted_lines += 1
                    
                    # Show progress every 100 lines
                    if converted_lines % 100 == 0:
                        print(f"  Processed {converted_lines} lines...", end='\r')
                
                except json.JSONDecodeError as e:
                    print(f"\n⚠️  Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                    # Write original line if parsing fails
                    fout.write(line + '\n')
                    failed_lines += 1
        
        print(f"\n✅ Conversion complete!")
        print(f"   Converted: {converted_lines} lines")
        if failed_lines > 0:
            print(f"   Failed: {failed_lines} lines (written as-is)")
        print(f"   Output: {output_file}")
        
        # Show preview
        print("\n" + "="*80)
        print("Preview (first 30 lines):")
        print("="*80)
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 30:
                    print("... (truncated)")
                    break
                print(line, end='')
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Missing input file argument", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    convert_instances_log(input_file, output_file)


if __name__ == "__main__":
    main()






















