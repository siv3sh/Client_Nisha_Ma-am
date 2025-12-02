# Architecture Diagrams - PNG Files

This directory contains PNG renderings of all Mermaid diagrams from `ARCHITECTURE.md`.

## Generated Diagrams

1. **01_system_architecture.png** - System Architecture Overview
   - Shows frontend, document processing, RAG pipeline, LLM layer, and features
   - Highlights where RAG is used in the system

2. **02_rag_pipeline_flow.png** - RAG Pipeline Flow
   - Complete flow from document upload to answer generation
   - Shows document processing, embedding, storage, retrieval, reranking, and LLM generation

3. **03_component_interaction.png** - Component Interaction Flow (Sequence Diagram)
   - Step-by-step interaction between User, Streamlit, Processor, RAG Pipeline, Qdrant, and Gemini API

4. **04_feature_architecture.png** - Feature Architecture
   - Shows core features, advanced features, user features, and quality features
   - Demonstrates how all features connect to the RAG pipeline

5. **05_data_flow.png** - Data Flow Diagram
   - Shows data transformation from input documents through processing, storage, retrieval, to final answer

6. **06_technology_stack.png** - Technology Stack
   - Visual representation of all technologies used in the system

## Usage

These PNG files can be:
- Included in presentations
- Added to documentation
- Shared with stakeholders
- Used in README files (GitHub supports both Mermaid and images)

## Regenerating Diagrams

To regenerate these PNG files, run:

```bash
python3 generate_diagrams.py
```

This script extracts Mermaid code from `ARCHITECTURE.md` and converts each diagram to PNG using the mermaid.ink API.

## Notes

- Diagrams are generated at standard resolution
- For higher resolution, you can use mermaid-cli (mmdc) locally
- Original Mermaid source code is available in `ARCHITECTURE.md`

