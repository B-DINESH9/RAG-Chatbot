"""Tests for retriever module."""
import pytest


def test_create_base_retriever(tmp_dir, sample_documents):
    """Test creating a base vectorstore retriever."""
    from src.embeddings import get_embeddings
    from src.vector_store import create_vector_store
    from src.retriever import create_retriever
    from src.text_splitter import split_documents

    embeddings = get_embeddings("local")
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    persist_dir = str(tmp_dir / "test_store")
    vector_store = create_vector_store(chunks, embeddings, persist_dir)

    retriever = create_retriever(
        vector_store=vector_store,
        embeddings=embeddings,
        retriever_type="Vectorstore backed retriever",
        base_retriever_search_type="similarity",
        base_retriever_k=4,
    )
    assert retriever is not None


def test_create_compression_retriever(tmp_dir, sample_documents):
    """Test creating a contextual compression retriever."""
    from src.embeddings import get_embeddings
    from src.vector_store import create_vector_store
    from src.retriever import create_retriever
    from src.text_splitter import split_documents

    embeddings = get_embeddings("local")
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    persist_dir = str(tmp_dir / "test_store")
    vector_store = create_vector_store(chunks, embeddings, persist_dir)

    retriever = create_retriever(
        vector_store=vector_store,
        embeddings=embeddings,
        retriever_type="Contextual compression",
        base_retriever_search_type="similarity",
        base_retriever_k=4,
        compression_retriever_k=4,
    )
    assert retriever is not None


def test_create_retriever_invalid_type(tmp_dir, sample_documents):
    """Test that invalid retriever type raises ValueError."""
    from src.embeddings import get_embeddings
    from src.vector_store import create_vector_store
    from src.retriever import create_retriever
    from src.text_splitter import split_documents

    embeddings = get_embeddings("local")
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    persist_dir = str(tmp_dir / "test_store")
    vector_store = create_vector_store(chunks, embeddings, persist_dir)

    with pytest.raises(ValueError):
        create_retriever(
            vector_store=vector_store,
            embeddings=embeddings,
            retriever_type="invalid_type",
        )
