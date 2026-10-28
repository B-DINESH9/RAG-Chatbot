"""Tests for configuration module."""
from pathlib import Path


def test_directories_exist():
    """Test that required directories are defined."""
    from src.config import TMP_DIR, VECTOR_STORE_DIR, BASE_DIR
    assert isinstance(TMP_DIR, Path)
    assert isinstance(VECTOR_STORE_DIR, Path)
    assert isinstance(BASE_DIR, Path)


def test_llm_providers_defined():
    """Test that LLM providers list is populated."""
    from src.config import LLM_PROVIDERS
    assert len(LLM_PROVIDERS) >= 3
    assert "OpenAI" in LLM_PROVIDERS
    assert "Google" in LLM_PROVIDERS
    assert "HuggingFace" in LLM_PROVIDERS


def test_models_defined():
    """Test that model lists are populated."""
    from src.config import OPENAI_MODELS, GOOGLE_MODELS, HUGGINGFACE_MODELS
    assert len(OPENAI_MODELS) > 0
    assert len(GOOGLE_MODELS) > 0
    assert len(HUGGINGFACE_MODELS) > 0


def test_retriever_types_defined():
    """Test that retriever types are defined."""
    from src.config import RETRIEVER_TYPES
    assert len(RETRIEVER_TYPES) == 3


def test_welcome_messages_complete():
    """Test that welcome messages cover all languages."""
    from src.config import WELCOME_MESSAGES
    expected_languages = ["english", "french", "spanish", "german", "russian", 
                         "chinese", "arabic", "portuguese", "italian", "japanese"]
    for lang in expected_languages:
        assert lang in WELCOME_MESSAGES, f"Missing welcome message for {lang}"
        assert len(WELCOME_MESSAGES[lang]) > 0


def test_default_parameters():
    """Test default chunking parameters."""
    from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    assert DEFAULT_CHUNK_SIZE > 0
    assert DEFAULT_CHUNK_OVERLAP >= 0
    assert DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE
