#!/usr/bin/env python3
"""
Fix Mermaid syntax by removing HTML tags and fixing syntax issues
"""

import re

def fix_mermaid_diagram(code):
    """Fix common Mermaid syntax issues."""
    # Replace <br/> with line breaks or spaces
    code = re.sub(r'<br/>', ' ', code)
    code = re.sub(r'<br\s*/?>', ' ', code)
    
    # Fix any other HTML entities
    code = code.replace('&amp;', '&')
    code = code.replace('&lt;', '<')
    code = code.replace('&gt;', '>')
    
    # Remove extra whitespace
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        # Replace multiple spaces with single space
        line = re.sub(r' +', ' ', line)
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def fix_architecture_file():
    """Fix all Mermaid diagrams in ARCHITECTURE.md"""
    with open('ARCHITECTURE.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and fix all mermaid blocks
    pattern = r'(```mermaid\n)(.*?)(```)'
    
    def replace_func(match):
        prefix = match.group(1)
        diagram_code = match.group(2)
        suffix = match.group(3)
        fixed_code = fix_mermaid_diagram(diagram_code)
        return prefix + fixed_code + suffix
    
    fixed_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    with open('ARCHITECTURE.md', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Fixed Mermaid diagrams in ARCHITECTURE.md")

if __name__ == "__main__":
    fix_architecture_file()

