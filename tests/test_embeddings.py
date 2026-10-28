"""Tests for embeddings module."""
import pytest


def test_get_embeddings_unknown_provider():
    """Test that unknown provider raises ValueError."""
    from src.embeddings import get_embeddings
    with pytest.raises(ValueError):
        get_embeddings("UnknownProvider", api_key="test")


def test_get_embeddings_local():
    """Test getting local embeddings (sentence-transformers)."""
    from src.embeddings import get_embeddings
    embeddings = get_embeddings("local")
    assert embeddings is not None
    # Verify it can embed a simple text
    result = embeddings.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_embeddings_openai_without_key():
    """Test OpenAI embeddings instantiation (will fail at API call time, not creation)."""
    from src.embeddings import get_embeddings
    # OpenAI embeddings can be created without key, fails on use
    embeddings = get_embeddings("OpenAI", api_key="sk-test-fake-key")
    assert embeddings is not None
