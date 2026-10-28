"""Conversation memory module."""
from langchain_classic.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain_classic.base_memory import BaseMemory

def create_memory(model_name: str = "gpt-3.5-turbo", openai_api_key: str = None, memory_max_token: int = None) -> BaseMemory:
    """Create a memory for the conversation chain."""
    if model_name == "gpt-3.5-turbo" and openai_api_key:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            openai_api_key=openai_api_key, 
            temperature=0.1
        )
        return ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=memory_max_token or 1024,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
    else:
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
