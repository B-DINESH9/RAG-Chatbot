"""Tests for chain module."""
import pytest


def test_get_answer_template():
    """Test answer template generation."""
    from src.chain import get_answer_template
    template = get_answer_template("english")
    assert "{context}" in template
    assert "{question}" in template
    assert "{chat_history}" in template
    assert "english" in template.lower()


def test_get_answer_template_french():
    """Test answer template with French language."""
    from src.chain import get_answer_template
    template = get_answer_template("french")
    assert "french" in template.lower()


def test_get_condense_prompt():
    """Test condense question prompt."""
    from src.chain import get_condense_prompt
    prompt = get_condense_prompt()
    assert prompt is not None
    assert "chat_history" in prompt.input_variables
    assert "question" in prompt.input_variables


def test_create_llm_invalid_provider():
    """Test that invalid LLM provider raises ValueError."""
    from src.chain import create_llm
    with pytest.raises(ValueError):
        create_llm("InvalidProvider", "fake-key", "model")
