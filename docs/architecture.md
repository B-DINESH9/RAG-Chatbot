# Architecture Documentation

## System Architecture

```mermaid
graph TB
    subgraph "🖥️ User Interface Layer"
        UI["Streamlit Web App<br/>Port 8501"]
        SIDE["Sidebar Controls<br/>Provider • Model • Language"]
        CHAT["Chat Interface<br/>Conversational Q&A"]
    end

    subgraph "📄 Document Ingestion Pipeline"
        UL["File Upload<br/>PDF / DOCX / CSV / TXT"]
        DL["Document Loaders<br/>PyPDF • Docx2txt • CSV • Text"]
        CS["Text Splitter<br/>RecursiveCharacterTextSplitter<br/>chunk_size=1600, overlap=200"]
        EMB["Embedding Model<br/>sentence-transformers/all-MiniLM-L6-v2<br/>OpenAI / Google / HuggingFace"]
    end

    subgraph "💾 Vector Storage Layer"
        VDB["ChromaDB<br/>Persistent Vector Store"]
    end

    subgraph "🔍 Retrieval & Compression Layer"
        BR["Base Retriever<br/>Similarity / MMR Search"]
        CC["Contextual Compression<br/>Split → Deduplicate → Filter → Reorder"]
        CR["Cohere Reranker<br/>rerank-multilingual-v2.0"]
    end

    subgraph "🤖 LLM Response Layer"
        SQ["Condense Question LLM<br/>Follow-up → Standalone Query"]
        MEM["Conversation Memory<br/>BufferMemory / SummaryBufferMemory"]
        LLM["Response Generation LLM<br/>GPT-4 / Gemini Pro / Mistral-7B"]
    end

    subgraph "☁️ Infrastructure Layer"
        DOCKER["Docker Container<br/>Multi-stage Build"]
        GCR["Google Cloud Run<br/>Serverless Deployment"]
        HC["Health Check<br/>/_stcore/health"]
    end

    UI --> SIDE
    UI --> CHAT
    UI --> UL
    UL --> DL --> CS --> EMB --> VDB
    CHAT -->|"User Query"| BR
    VDB --> BR
    BR --> CC
    BR --> CR
    CC --> SQ
    CR --> SQ
    SQ --> MEM --> LLM
    LLM -->|"Answer + Sources"| CHAT
    DOCKER --> GCR
    GCR --> HC
```

## Data Flow: Query Processing

```mermaid
sequenceDiagram
    participant U as User
    participant ST as Streamlit UI
    participant CQ as Condense Question LLM
    participant R as Retriever
    participant VDB as ChromaDB
    participant LLM as Response LLM
    participant M as Memory

    U->>ST: Enter question
    ST->>M: Get chat history
    M-->>ST: Previous conversation
    ST->>CQ: Question + Chat History
    CQ-->>ST: Standalone question
    ST->>R: Standalone question
    R->>VDB: Similarity search
    VDB-->>R: Top-k relevant chunks
    R-->>ST: Retrieved documents
    ST->>LLM: Standalone question + Context + History
    LLM-->>ST: Generated answer
    ST->>M: Store Q&A pair
    ST->>U: Display answer + source documents
```

## Module Dependency Graph

```mermaid
graph LR
    APP["app.py"] --> CONFIG["config.py"]
    APP --> DL["document_loader.py"]
    APP --> TS["text_splitter.py"]
    APP --> EMB["embeddings.py"]
    APP --> VS["vector_store.py"]
    APP --> RET["retriever.py"]
    APP --> CH["chain.py"]

    CH --> MEM["memory.py"]
    CH --> EMB
    RET --> VS
    DL --> CONFIG
    VS --> CONFIG

    style APP fill:#4CAF50,color:#fff
    style CONFIG fill:#2196F3,color:#fff
    style CH fill:#FF9800,color:#fff
    style MEM fill:#FF9800,color:#fff
```

## Retriever Strategies

### 1. Vectorstore Backed Retriever (Base)
Simple similarity search against the ChromaDB vector store. Returns top-k most similar document chunks.

### 2. Contextual Compression
A multi-stage pipeline that improves retrieval quality:
1. **CharacterTextSplitter** — Re-split retrieved docs into smaller chunks (500 chars)
2. **EmbeddingsRedundantFilter** — Remove duplicate/near-duplicate chunks
3. **EmbeddingsFilter** — Keep only chunks above relevance threshold
4. **LongContextReorder** — Place most relevant chunks at beginning/end (Lost in the Middle phenomenon)

### 3. Cohere Reranker
Uses Cohere's `rerank-multilingual-v2.0` model to re-score and reorder retrieved documents by relevance. Requires a Cohere API key.

## Docker Architecture

```mermaid
graph TB
    subgraph "Build Stage (python:3.11-slim)"
        B1["Install system deps"]
        B2["pip install requirements"]
    end

    subgraph "Runtime Stage (python:3.11-slim)"
        R1["Copy installed packages"]
        R2["Copy application code"]
        R3["Create data directories"]
        R4["Expose port 8501"]
        R5["Health check endpoint"]
        R6["Run Streamlit"]
    end

    B1 --> B2
    B2 -->|"COPY --from=builder"| R1
    R1 --> R2 --> R3 --> R4 --> R5 --> R6
```
