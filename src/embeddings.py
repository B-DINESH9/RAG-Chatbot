"""Embedding model selection."""
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings

def get_embeddings(provider: str, api_key: str = None) -> Embeddings:
    """Get embedding model based on provider."""
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return OpenAIEmbeddings(openai_api_key=api_key)
    elif provider_lower == "google":
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    elif provider_lower == "huggingface":
        return HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name="thenlper/gte-large")
    elif provider_lower in ["local", "groq"] or not provider:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
