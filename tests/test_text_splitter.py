"""Tests for text splitting module."""
import pytest


def test_split_documents(sample_documents):
    """Test basic document splitting."""
    from src.text_splitter import split_documents
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > len(sample_documents)


def test_split_empty_list():
    """Test splitting an empty list."""
    from src.text_splitter import split_documents
    chunks = split_documents([])
    assert len(chunks) == 0


def test_split_preserves_metadata(sample_documents):
    """Test that metadata is preserved after splitting."""
    from src.text_splitter import split_documents
    chunks = split_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    for chunk in chunks:
        assert "source" in chunk.metadata


def test_split_invalid_chunk_size():
    """Test that invalid chunk size raises ValueError."""
    from src.text_splitter import split_documents
    with pytest.raises(ValueError):
        split_documents([], chunk_size=-1)


def test_split_invalid_overlap():
    """Test that overlap >= chunk_size raises ValueError."""
    from src.text_splitter import split_documents
    with pytest.raises(ValueError):
        split_documents([], chunk_size=100, chunk_overlap=100)
