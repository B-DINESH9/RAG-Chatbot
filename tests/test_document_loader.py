"""Tests for document loading module."""
import os
from pathlib import Path

import pytest


def test_load_txt_documents(tmp_dir, sample_txt_file):
    """Test loading text files."""
    from src.document_loader import load_documents
    documents = load_documents(tmp_dir)
    assert len(documents) > 0
    assert "RAG" in documents[0].page_content


def test_load_csv_documents(tmp_dir, sample_csv_file):
    """Test loading CSV files."""
    from src.document_loader import load_documents
    documents = load_documents(tmp_dir)
    assert len(documents) > 0


def test_load_empty_directory(tmp_dir):
    """Test loading from an empty directory."""
    from src.document_loader import load_documents
    documents = load_documents(tmp_dir)
    assert len(documents) == 0


def test_delete_temp_files(tmp_dir, sample_txt_file):
    """Test deleting temporary files."""
    from src.document_loader import delete_temp_files
    assert sample_txt_file.exists()
    delete_temp_files(tmp_dir)
    assert not sample_txt_file.exists()


def test_delete_temp_files_empty_dir(tmp_dir):
    """Test deleting temp files in empty directory doesn't error."""
    from src.document_loader import delete_temp_files
    delete_temp_files(tmp_dir)  # Should not raise
