"""Tests for vector store module."""
import shutil
from pathlib import Path

import pytest


def test_create_vector_store(tmp_dir, sample_documents):
    """Test creating a ChromaDB vector store."""
    from src.embeddings import get_embeddings
    from src.vector_store import create_vector_store
    from src.text_splitter import split_documents

    embeddings = get_embeddings("local")
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    persist_dir = str(tmp_dir / "test_store")
    
    vector_store = create_vector_store(chunks, embeddings, persist_dir)
    assert vector_store is not None


def test_load_vector_store(tmp_dir, sample_documents):
    """Test loading an existing vector store."""
    from src.embeddings import get_embeddings
    from src.vector_store import create_vector_store, load_vector_store
    from src.text_splitter import split_documents

    embeddings = get_embeddings("local")
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    persist_dir = str(tmp_dir / "test_store")
    
    create_vector_store(chunks, embeddings, persist_dir)
    loaded = load_vector_store(embeddings, persist_dir)
    assert loaded is not None


def test_list_vector_stores(tmp_dir):
    """Test listing vector stores."""
    from src.vector_store import list_vector_stores
    
    # Create some fake store directories
    (tmp_dir / "store1").mkdir()
    (tmp_dir / "store2").mkdir()
    (tmp_dir / "not_a_file.txt").write_text("test")
    
    stores = list_vector_stores(tmp_dir)
    assert "store1" in stores
    assert "store2" in stores
    assert "not_a_file.txt" not in stores


def test_list_vector_stores_empty(tmp_dir):
    """Test listing vector stores in empty directory."""
    from src.vector_store import list_vector_stores
    stores = list_vector_stores(tmp_dir)
    assert len(stores) == 0
