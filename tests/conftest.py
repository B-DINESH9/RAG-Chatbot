"""Shared test fixtures for RAG Chatbot test suite."""
import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_txt_file(tmp_dir):
    """Create a sample text file."""
    file_path = tmp_dir / "test.txt"
    file_path.write_text(
        "This is a test document about RAG systems. "
        "RAG combines retrieval and generation for better answers. "
        "It uses vector databases to find relevant context.",
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def sample_csv_file(tmp_dir):
    """Create a sample CSV file."""
    file_path = tmp_dir / "test.csv"
    file_path.write_text(
        "question,answer\n"
        "What is RAG?,Retrieval-Augmented Generation\n"
        "What is an embedding?,A vector representation of text\n",
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def sample_documents():
    """Create sample LangChain Document objects."""
    from langchain.schema import Document
    return [
        Document(page_content="RAG systems use retrieval to augment generation. " * 20, metadata={"source": "test1.txt"}),
        Document(page_content="Vector databases store embeddings for semantic search. " * 20, metadata={"source": "test2.txt"}),
        Document(page_content="Large language models generate human-like text responses. " * 20, metadata={"source": "test3.txt"}),
    ]


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings object for testing."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.embed_documents.return_value = [[0.1] * 384] * 3
    mock.embed_query.return_value = [0.1] * 384
    return mock
