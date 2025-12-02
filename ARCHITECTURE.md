# 🏗️ Architecture & Flow Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph Frontend["Frontend Layer"]
        UI[Streamlit Web Interface]
        Sidebar[Sidebar Controls]
        Chat[Chat Interface]
    end

    subgraph Processing["Document Processing Layer"]
        Upload[Document Upload]
        OCR[Tesseract OCR + PaddleOCR]
        Extract[Text Extraction]
        LangDetect[Language Detection]
        Chunking[Text Chunking]
    end

    subgraph RAG["RAG Pipeline Core"]
        Embedding[Multilingual Embeddings]
        VectorDB[(Qdrant Vector Database)]
        Retrieval[Semantic Retrieval]
        Reranker[Cross-Encoder Reranking]
    end

    subgraph LLM["LLM Layer"]
        Gemini[Google Gemini API]
        Prompt[Prompt Engineering]
        Translation[Translation Service]
    end

    subgraph Features["Features & Utilities"]
        Glossary[Glossary Extraction]
        Export[Export & Share]
        Filters[Retrieval Filters]
    end

    UI --> Sidebar
    UI --> Chat
    Sidebar --> Upload
    Upload --> OCR
    Upload --> Extract
    Extract --> LangDetect
    LangDetect --> Chunking
    Chunking --> Embedding
    Embedding --> VectorDB
    
    Chat --> Retrieval
    Retrieval --> VectorDB
    Retrieval --> Reranker
    Reranker --> Prompt
    Prompt --> Gemini
    Gemini --> Chat
    
    Chat --> Translation
    Chat --> Export
    Chunking --> Glossary
    Retrieval --> Filters
    
    style VectorDB fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style Gemini fill:#4285F4,stroke:#1976D2,stroke-width:3px,color:#fff
    style Embedding fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    style Reranker fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
```

## RAG Pipeline Flow

```mermaid
flowchart TD
    Start([User Uploads Document]) --> ExtractText{Extract Text}
    
    ExtractText -->|PDF| PDFExtract[pdfplumber]
    ExtractText -->|DOCX| DOCXExtract[docx2txt]
    ExtractText -->|Image| OCRProcess[Tesseract + PaddleOCR]
    
    PDFExtract --> LangDetect[Detect Language]
    DOCXExtract --> LangDetect
    OCRProcess --> LangDetect
    
    LangDetect --> Chunk[Chunk Text]
    
    Chunk --> Embed[Generate Embeddings]
    
    Embed --> Store[(Store in Qdrant)]
    
    Store --> Ready([Document Ready])
    
    Ready --> Query([User Asks Question])
    
    Query --> AutoLang{Auto-detect Query Language?}
    AutoLang -->|Yes| LangFilter[Apply Language Filter]
    AutoLang -->|No| AllLangs[Search All Languages]
    
    LangFilter --> Variants[Generate Query Variants]
    AllLangs --> Variants
    
    Variants --> Retrieve[Semantic Search]
    
    Retrieve --> Filter[Post-Retrieval Filtering]
    
    Filter --> Rerank[Rerank Results]
    
    Rerank --> Context[Format Context Chunks]
    
    Context --> Prompt[Construct Prompt]
    
    Prompt --> LLM[Gemini API Generate Answer]
    
    LLM --> Answer[Return Answer in Query Language]
    
    Answer --> Verify[Source Verification]
    
    Verify --> Translate{Non-English Answer?}
    
    Translate -->|Yes| TransBtn[Show Translation Button]
    Translate -->|No| Display[Display Answer]
    
    TransBtn --> Display
    Display --> Export[Export Options]
    
    Export --> End([End])
    
    style Store fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style LLM fill:#4285F4,stroke:#1976D2,stroke-width:3px,color:#fff
    style Embed fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    style Rerank fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
```

## Component Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit UI
    participant Processor as Document Processor
    participant RAG as RAG Pipeline
    participant Qdrant as Vector DB
    participant Gemini as Gemini API
    
    User->>Streamlit: Upload Document
    Streamlit->>Processor: Process Document
    Processor->>Processor: Extract Text PDF/DOCX/OCR
    Processor->>Processor: Detect Language
    Processor->>Processor: Chunk Text
    Processor->>RAG: Generate Embeddings
    RAG->>Qdrant: Store Vectors + Metadata
    
    User->>Streamlit: Ask Question
    Streamlit->>Streamlit: Auto-detect Query Language
    Streamlit->>RAG: Retrieve Context
    RAG->>Qdrant: Semantic Search
    Qdrant-->>RAG: Return Top-K Chunks
    RAG->>RAG: Rerank Results
    RAG-->>Streamlit: Context Chunks
    
    Streamlit->>Gemini: Generate Answer with context + prompt
    Gemini-->>Streamlit: Answer in Query Language
    Streamlit->>User: Display Answer + Sources
    
    alt Translation Requested
        User->>Streamlit: Click Translate
        Streamlit->>Gemini: Translate to English
        Gemini-->>Streamlit: English Translation
        Streamlit->>User: Show Translation
    end
```

## Feature Architecture

```mermaid
graph LR
    subgraph Core["Core Features"]
        A[Multilingual Support]
        B[RAG Pipeline]
        C[Vector Search]
    end
    
    subgraph Advanced["Advanced Features"]
        D[OCR Processing]
        E[Language Detection]
        F[Transliteration]
        G[Reranking]
    end
    
    subgraph User["User Features"]
        H[Batch Upload]
        I[Retrieval Filters]
        J[Answer Modes]
        K[Translation]
        L[Export & Share]
        M[Glossary]
    end
    
    subgraph Quality["Quality Features"]
        N[Source Verification]
        O[Duplicate Detection]
        P[MMR Diversity]
        Q[Error Handling]
    end
    
    A --> B
    B --> C
    D --> B
    E --> B
    F --> B
    G --> B
    B --> H
    B --> I
    B --> J
    B --> K
    B --> L
    B --> M
    B --> N
    B --> O
    B --> P
    B --> Q
    
    style B fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style C fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    style G fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input["Input"]
        Doc[Documents PDF/DOCX/TXT/Images]
        Query[User Query Multilingual]
    end
    
    subgraph Processing["Processing"]
        Text[Text Extraction]
        Chunks[Text Chunks with Metadata]
        Embed[Embeddings 768-dim vectors]
    end
    
    subgraph Storage["Storage"]
        Vectors[(Qdrant Vector Store)]
        Meta[(Metadata)]
    end
    
    subgraph Retrieval["Retrieval"]
        Search[Semantic Search]
        Rank[Reranking]
        Context[Context Chunks]
    end
    
    subgraph Generation["Generation"]
        Prompt[Prompt Construction]
        LLM[Gemini LLM]
        Answer[Multilingual Answer]
    end
    
    Doc --> Text
    Text --> Chunks
    Chunks --> Embed
    Embed --> Vectors
    Chunks --> Meta
    
    Query --> Search
    Vectors --> Search
    Meta --> Search
    Search --> Rank
    Rank --> Context
    Context --> Prompt
    Query --> Prompt
    Prompt --> LLM
    LLM --> Answer
    
    style Vectors fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style LLM fill:#4285F4,stroke:#1976D2,stroke-width:3px,color:#fff
    style Embed fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
```

## Technology Stack

```mermaid
graph TB
    subgraph Frontend["Frontend"]
        S[Streamlit Web Framework]
    end
    
    subgraph Backend["Backend Processing"]
        P[Python 3.8+]
        T[Tesseract OCR]
        PD[PaddleOCR]
        PL[pdfplumber]
        D2T[docx2txt]
    end
    
    subgraph ML["ML/NLP"]
        ST[sentence-transformers Embeddings]
        TR[transformers Reranking]
        LD[langdetect Language Detection]
        IT[indic-transliteration]
    end
    
    subgraph VectorDB["Vector Database"]
        Q[Qdrant Vector Store]
    end
    
    subgraph LLMAPI["LLM API"]
        G[Google Gemini API gemini-2.5-flash]
    end
    
    S --> P
    P --> T
    P --> PD
    P --> PL
    P --> D2T
    P --> ST
    P --> TR
    P --> LD
    P --> IT
    ST --> Q
    TR --> Q
    P --> G
    
    style Q fill:#4CAF50,stroke:#2E7D32,stroke-width:3px,color:#fff
    style G fill:#4285F4,stroke:#1976D2,stroke-width:3px,color:#fff
    style ST fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
```

## Key Features Highlight

### 1. **RAG (Retrieval-Augmented Generation)**
- **Location**: Core of the system
- **Components**:
  - Document embedding generation
  - Vector storage in Qdrant
  - Semantic retrieval
  - Context augmentation for LLM
- **Benefits**: Grounds answers in uploaded documents, reduces hallucinations

### 2. **Multilingual Support**
- **Languages**: Malayalam, Tamil, Telugu, Kannada, Tulu, English
- **Features**:
  - Language detection
  - Multilingual embeddings (multilingual-e5-large)
  - Language-specific filtering
  - Translation to English

### 3. **Advanced Retrieval**
- **Reranking**: Cross-encoder (bge-reranker-large) improves relevance
- **MMR**: Maximal Marginal Relevance for diverse results
- **Filters**: Language, source, date range
- **Post-filtering**: Removes cross-language contamination

### 4. **OCR Capabilities**
- **Tesseract**: Primary OCR engine
- **PaddleOCR**: Fallback for complex layouts
- **Preprocessing**: Denoising, thresholding, contrast enhancement

### 5. **Smart Chunking**
- Dynamic chunk sizes based on text length
- Overlap handling to preserve context
- Duplicate detection using word-level similarity

### 6. **User Experience**
- Batch document upload
- Multiple answer formats (concise, detailed, bullet, step-by-step)
- Source verification with expandable context
- Export and sharing capabilities
- Auto-extracted glossary

---

## How to View These Diagrams

1. **GitHub**: These Mermaid diagrams render automatically in GitHub markdown
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Copy diagram code to [Mermaid Live Editor](https://mermaid.live/)
4. **Documentation**: Include in README.md or documentation site

---

## Architecture Highlights

- **Scalable**: Vector database supports millions of documents
- **Multilingual**: Native support for 5+ South Indian languages
- **Accurate**: RAG + Reranking ensures relevant context
- **User-Friendly**: Intuitive UI with helpful features
- **Robust**: Multiple OCR engines and error handling
- **Flexible**: Configurable filters and answer modes

