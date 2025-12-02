#!/usr/bin/env python3
"""
Generate PNG images from Mermaid diagrams in ARCHITECTURE.md
Uses mermaid.ink API to convert Mermaid code to PNG
"""

import re
import requests
import os
from pathlib import Path

def extract_mermaid_diagrams(markdown_file):
    """Extract all Mermaid diagram code blocks from markdown file."""
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all mermaid code blocks
    pattern = r'```mermaid\n(.*?)```'
    matches = re.findall(pattern, content, re.DOTALL)
    
    diagrams = []
    for i, diagram_code in enumerate(matches, 1):
        diagrams.append({
            'index': i,
            'code': diagram_code.strip()
        })
    
    return diagrams

def generate_png_from_mermaid(mermaid_code, output_path):
    """Generate PNG from Mermaid code using mermaid.ink API."""
    import base64
    import zlib
    import json
    
    # Clean the mermaid code - remove any problematic characters
    cleaned_code = mermaid_code.replace('→', 'to').replace('→', '->')
    
    # Try the new mermaid.ink API format (JSON-based)
    try:
        # Method 1: Try JSON API endpoint
        api_url = "https://mermaid.ink/api/v2/png"
        payload = {
            "code": cleaned_code,
            "mermaid": {"theme": "default"}
        }
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e1:
        print(f"JSON API failed: {e1}, trying legacy method...")
    
    # Method 2: Legacy base64 method
    try:
        mermaid_bytes = cleaned_code.encode('utf-8')
        compressed = zlib.compress(mermaid_bytes)
        encoded = base64.urlsafe_b64encode(compressed).decode('ascii').rstrip('=')
        url = f"https://mermaid.ink/img/{encoded}"
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # Check if it's actually an image
            content_type = response.headers.get('Content-Type', '')
            if 'image' in content_type or len(response.content) > 1000:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"Warning: Response doesn't look like an image (size: {len(response.content)} bytes)")
                return False
        else:
            print(f"Error: API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Error generating PNG: {e}")
        return False

def main():
    markdown_file = "ARCHITECTURE.md"
    output_dir = "diagrams"
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Extract diagrams
    diagrams = extract_mermaid_diagrams(markdown_file)
    
    if not diagrams:
        print("No Mermaid diagrams found in ARCHITECTURE.md")
        return
    
    print(f"Found {len(diagrams)} Mermaid diagrams")
    
    # Diagram names based on their order
    diagram_names = [
        "01_system_architecture",
        "02_rag_pipeline_flow",
        "03_component_interaction",
        "04_feature_architecture",
        "05_data_flow",
        "06_technology_stack"
    ]
    
    # Generate PNGs
    for i, diagram in enumerate(diagrams):
        name = diagram_names[i] if i < len(diagram_names) else f"diagram_{diagram['index']}"
        output_path = os.path.join(output_dir, f"{name}.png")
        
        print(f"Generating {output_path}...")
        if generate_png_from_mermaid(diagram['code'], output_path):
            print(f"✅ Created {output_path}")
        else:
            print(f"❌ Failed to create {output_path}")
    
    print(f"\n✅ All diagrams generated in '{output_dir}' directory")

if __name__ == "__main__":
    main()

