"""ChromaDB operations."""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

def create_vector_store(chunks: List[Document], embeddings: Embeddings, vectorstore_name: str, base_dir: Path) -> Chroma:
    """Create a new Chroma vector store from documents."""
    persist_directory = str(base_dir / vectorstore_name)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

def load_vector_store(vectorstore_name: str, embeddings: Embeddings, base_dir: Path) -> Chroma:
    """Load an existing Chroma vector store."""
    persist_directory = str(base_dir / vectorstore_name)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

def list_vector_stores(base_dir: Path) -> List[str]:
    """List available vector stores in a directory."""
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    
    stores = []
    for item in base_dir.iterdir():
        if item.is_dir():
            # Basic check if it looks like a Chroma DB (has sqlite3 or other DB files)
            # We'll just assume any directory inside the vector store base dir is a store
            stores.append(item.name)
            
    return stores
