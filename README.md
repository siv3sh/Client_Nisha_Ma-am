# Multilingual Document QA Chatbot

A production-ready Retrieval-Augmented Generation (RAG) system for querying documents in South Indian languages (Malayalam, Tamil, Telugu, Kannada, Tulu) and English. Built with Streamlit, Google Gemini API, Qdrant vector database, and advanced multilingual embeddings.

## Overview

This application enables users to upload documents in multiple formats (PDF, DOCX, TXT, Images) and ask questions in their native language. The system uses RAG (Retrieval-Augmented Generation) to retrieve relevant context from uploaded documents and generate accurate, language-specific answers using Google Gemini API.

## Key Features

- **Multilingual Support**: Native support for Malayalam, Tamil, Telugu, Kannada, Tulu, and English
- **RAG Pipeline**: Advanced retrieval-augmented generation for accurate, document-grounded answers
- **Batch Document Processing**: Upload and process multiple documents simultaneously
- **Intelligent Retrieval**: Semantic search with reranking, language filtering, and source filtering
- **OCR Capabilities**: Dual OCR engines (Tesseract + PaddleOCR) for scanned documents and images
- **Source Verification**: View exact documents and context chunks used for each answer
- **Translation**: Translate non-English answers to English on demand
- **Export & Share**: Export conversations as Markdown and generate shareable read-only links
- **Glossary Extraction**: Automatically extract and organize key terms from documents

## System Architecture

The system is built on a RAG (Retrieval-Augmented Generation) architecture that combines semantic search with large language model generation.

### Architecture Diagram

![System Architecture](diagrams/01_system_architecture.png)

The architecture consists of four main layers:

1. **Frontend Layer**: Streamlit web interface with sidebar controls and chat interface
2. **Document Processing Layer**: Text extraction, OCR, language detection, and intelligent chunking
3. **RAG Pipeline Core**: Multilingual embeddings, vector storage (Qdrant), semantic retrieval, and cross-encoder reranking
4. **LLM Layer**: Google Gemini API integration with multilingual prompt engineering

## RAG Pipeline Flow

The RAG pipeline is the core of the system, ensuring answers are grounded in uploaded documents.

### Complete RAG Flow Diagram

![RAG Pipeline Flow](diagrams/02_rag_pipeline_flow.png)

### How RAG Works in This System

#### 1. Document Ingestion Phase

When a document is uploaded:

1. **Text Extraction**: 
   - PDF files use `pdfplumber` for text extraction
   - DOCX files use `docx2txt`
   - Images and scanned PDFs use OCR (Tesseract primary, PaddleOCR fallback)

2. **Language Detection**: 
   - Automatically detects document language using `langdetect`
   - Supports Malayalam, Tamil, Telugu, Kannada, Tulu, English

3. **Text Chunking**: 
   - Dynamic chunk sizes based on text length
   - Overlap handling to preserve context across chunks
   - Duplicate detection using word-level similarity

4. **Embedding Generation**: 
   - Uses `intfloat/multilingual-e5-large` model
   - Generates 768-dimensional vectors for each chunk
   - Applies model-specific prefixes ("query:", "passage:") for optimal performance

5. **Vector Storage**: 
   - Stores embeddings in Qdrant vector database
   - Includes metadata: language, filename, chunk index, timestamps
   - Enables fast semantic similarity search

#### 2. Query Processing Phase

When a user asks a question:

1. **Query Language Detection**: 
   - Auto-detects query language if not explicitly filtered
   - Prevents cross-language contamination

2. **Query Variant Generation**: 
   - Handles code-mixed queries (e.g., Tanglish)
   - Transliterates Latin script to native scripts when needed

3. **Semantic Retrieval**: 
   - Embeds query using multilingual-e5-large
   - Performs cosine similarity search in Qdrant
   - Uses Maximal Marginal Relevance (MMR) for diverse results
   - Applies filters: language, source documents, date range

4. **Post-Retrieval Filtering**: 
   - Removes chunks that don't match query language
   - Ensures only relevant content is used

5. **Reranking**: 
   - Uses `BAAI/bge-reranker-large` cross-encoder
   - Re-evaluates retrieved chunks for better relevance
   - Selects top 3 most relevant chunks

6. **Context Formatting**: 
   - Formats chunks with metadata (source, chunk index, relevance score)
   - Removes duplicates
   - Prepares context for LLM

#### 3. Answer Generation Phase

1. **Prompt Construction**: 
   - System prompt with language-specific instructions
   - Context chunks with source citations
   - User query
   - Answer mode selection (concise, detailed, bullet, step-by-step)

2. **LLM Generation**: 
   - Sends prompt to Google Gemini API (default: gemini-2.5-flash)
   - Generates answer in the query language
   - Ensures answer is based only on provided context

3. **Source Verification**: 
   - Displays which documents were used
   - Shows exact context chunks with relevance scores
   - Allows users to verify answer accuracy

### Component Interaction Flow

![Component Interaction](diagrams/03_component_interaction.png)

This sequence diagram shows the step-by-step interaction between components:

1. User uploads document → Streamlit → Document Processor
2. Processor extracts text, detects language, chunks text
3. RAG Pipeline generates embeddings → Stores in Qdrant
4. User asks question → Streamlit auto-detects language
5. RAG Pipeline retrieves context from Qdrant
6. Reranks results → Returns to Streamlit
7. Streamlit sends to Gemini API → Generates answer
8. Displays answer with source verification

## Feature Architecture

![Feature Architecture](diagrams/04_feature_architecture.png)

The system includes:

- **Core Features**: Multilingual support, RAG pipeline, vector search
- **Advanced Features**: OCR processing, language detection, transliteration, reranking
- **User Features**: Batch upload, retrieval filters, answer modes, translation, export/share, glossary
- **Quality Features**: Source verification, duplicate detection, MMR diversity, error handling

## Data Flow

![Data Flow](diagrams/05_data_flow.png)

Data flows through the system as follows:

1. **Input**: Documents (PDF/DOCX/TXT/Images) and User Queries (Multilingual)
2. **Processing**: Text extraction → Chunks with metadata → Embeddings (768-dim vectors)
3. **Storage**: Qdrant vector store and metadata storage
4. **Retrieval**: Semantic search → Reranking → Context chunks
5. **Generation**: Prompt construction → Gemini LLM → Multilingual answer

## Technology Stack

![Technology Stack](diagrams/06_technology_stack.png)

### Frontend
- **Streamlit**: Web framework for interactive UI

### Backend Processing
- **Python 3.8+**: Core programming language
- **Tesseract OCR**: Primary OCR engine
- **PaddleOCR**: Fallback OCR for complex layouts
- **pdfplumber**: PDF text extraction
- **docx2txt**: DOCX text extraction

### ML/NLP
- **sentence-transformers**: Multilingual embeddings (multilingual-e5-large)
- **transformers**: Cross-encoder reranking (bge-reranker-large)
- **langdetect**: Language detection
- **indic-transliteration**: Code-mixed query handling

### Vector Database
- **Qdrant**: High-performance vector database for semantic search

### LLM API
- **Google Gemini API**: Language model for answer generation (gemini-2.5-flash)

## Project Structure

```
Multilingual_DOC_BOT/
├── main.py                 # Streamlit UI, document processing, chat interface
├── rag_pipeline.py         # RAG pipeline: embeddings, Qdrant, retrieval, reranking
├── llm_handler.py          # Gemini API integration, prompt engineering
├── utils.py                 # OCR, text extraction, language detection, glossary
├── test_project.py         # Test suite for regression testing
├── requirements.txt         # Python dependencies
├── env.example             # Environment variable template
├── ARCHITECTURE.md         # Detailed architecture documentation
├── diagrams/               # Architecture diagram PNG files
│   ├── 01_system_architecture.png
│   ├── 02_rag_pipeline_flow.png
│   ├── 03_component_interaction.png
│   ├── 04_feature_architecture.png
│   ├── 05_data_flow.png
│   └── 06_technology_stack.png
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Qdrant vector database (local or cloud)
- Google Gemini API key
- Tesseract OCR (for image/scan support)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/siv3sh/Client_Nisha_Ma-am.git
cd Client_Nisha_Ma-am
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Set up Qdrant** (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

5. **Configure environment**:
```bash
cp env.example .env
# Edit .env and add your GEMINI_API_KEY
```

6. **Run the application**:
```bash
streamlit run main.py
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)
- `GEMINI_MODEL`: Model to use (default: gemini-2.5-flash)
- `QDRANT_URL`: Qdrant server URL (default: http://localhost:6333)
- `QDRANT_COLLECTION_NAME`: Collection name (default: multilingual_docs)
- `EMBEDDING_MODEL`: Embedding model (default: intfloat/multilingual-e5-large)
- `EMBEDDING_NORMALIZE`: Normalize embeddings (default: true)
- `RERANKER_MODEL`: Reranker model (default: BAAI/bge-reranker-large)

## Usage

### Uploading Documents

1. Click "Upload Documents" in the sidebar
2. Select one or more files (PDF, DOCX, TXT, Images)
3. Click "Process Documents"
4. Wait for processing to complete (progress bar shows status)

### Asking Questions

1. Enter your question in the chat input (supports multiple languages)
2. The system auto-detects query language
3. Retrieves relevant context from uploaded documents
4. Generates answer in the query language
5. View source documents and context chunks used

### Retrieval Filters

- **Language Filter**: Restrict search to specific languages
- **Source Filter**: Search only specific documents
- **Date Range**: Filter by document upload date

### Answer Modes

- **Concise**: Short, focused answers (3 sentences max)
- **Detailed with citations**: Comprehensive answers with source references
- **Bullet summary**: Organized bullet points
- **Step-by-step**: Numbered steps with explicit reasoning

### Translation

- For non-English answers, click "Translate to English" button
- Translation appears below the original answer
- Can hide/show translation as needed

### Export & Share

- Export selected Q&A pairs as Markdown
- Generate read-only share links for conversations
- Share links preserve conversation history without requiring API keys

## RAG Implementation Details

### Why RAG?

RAG (Retrieval-Augmented Generation) combines the benefits of:
- **Semantic Search**: Find relevant information quickly
- **LLM Generation**: Generate natural language answers
- **Document Grounding**: Answers are based on uploaded documents, reducing hallucinations

### RAG Components in This System

1. **Embedding Model**: `intfloat/multilingual-e5-large`
   - Supports 100+ languages including all South Indian languages
   - Generates high-quality semantic embeddings
   - Uses model-specific prefixes for optimal performance

2. **Vector Database**: Qdrant
   - Fast similarity search
   - Supports filtering by metadata
   - Scales to millions of documents

3. **Retrieval Strategy**: 
   - Initial semantic search with cosine similarity
   - MMR (Maximal Marginal Relevance) for diversity
   - Language and source filtering
   - Post-retrieval filtering for accuracy

4. **Reranking**: `BAAI/bge-reranker-large`
   - Cross-encoder model for better relevance
   - Re-evaluates top candidates
   - Improves answer quality significantly

5. **Context Augmentation**:
   - Formats retrieved chunks with metadata
   - Includes source citations
   - Limits to top 3 most relevant chunks
   - Removes duplicates

### RAG Flow Summary

1. **Document Upload** → Extract text → Detect language → Chunk → Embed → Store in Qdrant
2. **Query** → Detect language → Generate variants → Embed query → Search Qdrant → Filter → Rerank → Format context
3. **Generation** → Construct prompt with context → Send to Gemini → Generate answer → Verify sources

## Testing

Run the test suite:

```bash
python test_project.py
```

Tests cover:
- Import sanity checks
- Text processing (extraction, cleaning, chunking)
- RAG pipeline (embeddings, storage, retrieval)
- Reranking functionality
- Gemini API connectivity
- Multilingual answer generation

## Performance Considerations

- **Embedding Cache**: Embeddings are cached to avoid recomputation
- **Batch Processing**: Multiple documents processed in parallel
- **Chunk Optimization**: Dynamic chunk sizes based on text length
- **Reranking**: Applied only to top candidates for efficiency
- **Context Limiting**: Top 3 chunks used to reduce prompt size

## Security

- API keys stored in environment variables (never in code)
- `.env` file excluded from version control
- Share links use encoded tokens (no sensitive data in URLs)
- Read-only mode for shared conversations

## Troubleshooting

### Common Issues

1. **Qdrant Connection Error**: Ensure Qdrant is running on port 6333
2. **OCR Not Working**: Install Tesseract OCR system package
3. **Model Download Slow**: First run downloads large models (~2GB)
4. **API Key Error**: Verify GEMINI_API_KEY is set correctly

See `TROUBLESHOOTING.md` for detailed solutions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for educational and commercial use.

## Acknowledgments

- Google Gemini API for LLM capabilities
- Qdrant for vector database infrastructure
- Sentence Transformers for multilingual embeddings
- Streamlit for the web framework
- Open source OCR tools (Tesseract, PaddleOCR)

## Contact

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This system implements a production-ready RAG pipeline specifically designed for multilingual document question-answering. The architecture ensures accurate, document-grounded answers while supporting multiple South Indian languages natively.
