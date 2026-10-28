"""Central configuration module."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = BASE_DIR / "data" / "tmp"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_stores"

TMP_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

LLM_PROVIDERS = ["OpenAI", "Google", "HuggingFace", "Groq"]
OPENAI_MODELS = ["gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-4-turbo-preview"]
GOOGLE_MODELS = ["gemini-pro"]
HUGGINGFACE_MODELS = ["mistralai/Mistral-7B-Instruct-v0.2"]
GROQ_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

RETRIEVER_TYPES = ["Cohere reranker", "Contextual compression", "Vectorstore backed retriever"]
SUPPORTED_FILE_TYPES = ["pdf", "txt", "docx", "csv"]

DEFAULT_CHUNK_SIZE = 1600
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

WELCOME_MESSAGES = {
    "english": "Hello! How can I help you today?",
    "french": "Bonjour! Comment puis-je vous aider aujourd'hui?",
    "spanish": "¡Hola! ¿Cómo puedo ayudarte hoy?",
    "german": "Hallo! Wie kann ich Ihnen heute helfen?",
    "russian": "Здравствуйте! Чем я могу вам помочь сегодня?",
    "chinese": "你好！今天我能怎么帮你？",
    "arabic": "مرحباً! كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Olá! Como posso ajudar você hoje?",
    "italian": "Ciao! Come posso aiutarti oggi?",
    "japanese": "こんにちは！今日はどのようにお手伝いできますか？"
}
