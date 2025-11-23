#!/usr/bin/env python3
"""
Fix all lists in QMD file to have blank lines before them.
Rule: ALL lists (ordered and unordered) must have blank line before them,
      unless they are sublists (continuation of existing list).
"""

import re

def is_list_item(line):
    """Check if line is a list item (ordered or unordered)."""
    stripped = line.lstrip()
    # Unordered list
    if stripped.startswith('- '):
        return True
    # Ordered list (e.g., "1. ", "2. ")
    if re.match(r'^\d+\.\s', stripped):
        return True
    return False

def is_sublist(line, prev_line):
    """Check if this is a sublist (indented continuation)."""
    if not line.startswith(' ') and not line.startswith('\t'):
        return False  # Not indented, so not a sublist
    # Check if previous line was a list item
    return is_list_item(prev_line)

def is_header(line):
    """Check if line is a markdown header."""
    stripped = line.lstrip()
    return stripped.startswith('#')

def is_blank(line):
    """Check if line is blank or whitespace only."""
    return line.strip() == ''

def fix_list_formatting(input_file, output_file):
    """Add blank lines before all lists that need them."""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    i = 0
    fixes_made = 0

    while i < len(lines):
        current_line = lines[i]

        # Check if current line is a list item
        if is_list_item(current_line):
            # Check if we need to add a blank line
            if i > 0:
                prev_line = lines[i-1]

                # Don't add blank line if:
                # 1. Previous line is blank
                # 2. Previous line is a header
                # 3. This is a sublist (continuation of previous list)
                # 4. Previous line is also a list item (continuing same list)
                needs_blank_line = not (
                    is_blank(prev_line) or
                    is_header(prev_line) or
                    is_list_item(prev_line)
                )

                if needs_blank_line:
                    # Add a blank line before this list item
                    fixed_lines.append('\n')
                    fixes_made += 1
                    print(f"Line {i+1}: Added blank line before: {current_line.strip()[:60]}...")

        fixed_lines.append(current_line)
        i += 1

    # Write fixed content
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

    print(f"\nTotal fixes made: {fixes_made}")
    return fixes_made

if __name__ == '__main__':
    input_file = 'federated_learning_aggregation_comparison.qmd'
    output_file = 'federated_learning_aggregation_comparison.qmd'

    print("Fixing list formatting in paper...")
    print("=" * 70)

    fixes = fix_list_formatting(input_file, output_file)

    print("=" * 70)
    print(f"Done! Made {fixes} fixes.")
