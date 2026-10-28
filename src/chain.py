"""ConversationalRetrievalChain builder."""
from typing import Tuple
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from langchain_classic.base_memory import BaseMemory
from langchain_core.retrievers import BaseRetriever

from src.memory import create_memory

def get_answer_template(language: str = "english") -> str:
    """Get the prompt template for answering questions."""
    return f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Reply entirely in {language}.

Chat History:
{{chat_history}}

Context:
{{context}}

Question: {{question}}
Answer in {language}:"""

def get_condense_prompt() -> PromptTemplate:
    """Get the prompt template for condensing a question."""
    condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
    return PromptTemplate.from_template(condense_template)

def create_llm(provider: str, api_key: str, model: str, temperature: float = 0.5, top_p: float = 0.95):
    """Factory to create an LLM."""
    provider_lower = provider.lower()
    
    if provider_lower == "openai":
        return ChatOpenAI(
            model_name=model, 
            temperature=temperature, 
            model_kwargs={"top_p": top_p},
            openai_api_key=api_key
        )
    elif provider_lower == "google":
        return ChatGoogleGenerativeAI(
            model=model, 
            temperature=temperature, 
            top_p=top_p,
            google_api_key=api_key
        )
    elif provider_lower == "huggingface":
        return HuggingFaceHub(
            repo_id=model, 
            huggingfacehub_api_token=api_key,
            model_kwargs={"temperature": temperature, "top_p": top_p}
        )
    elif provider_lower == "groq":
        if ChatGroq is None:
            raise ImportError("langchain-groq is not installed. Please run `pip install langchain-groq` in your terminal.")
        return ChatGroq(
            model_name=model,
            temperature=temperature,
            groq_api_key=api_key,
            model_kwargs={"top_p": top_p}
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

def create_chain(
    retriever: BaseRetriever, 
    provider: str, 
    model: str, 
    api_key: str, 
    language: str = "english", 
    temperature: float = 0.5, 
    top_p: float = 0.95, 
    openai_api_key: str = None
) -> Tuple[ConversationalRetrievalChain, BaseMemory]:
    """Create the ConversationalRetrievalChain."""
    
    # Condense question LLM (low temperature for consistency)
    condense_question_llm = create_llm(
        provider=provider, 
        api_key=api_key, 
        model=model, 
        temperature=0.1, 
        top_p=0.95
    )
    
    # Response generation LLM
    response_generation_llm = create_llm(
        provider=provider, 
        api_key=api_key, 
        model=model, 
        temperature=temperature, 
        top_p=top_p
    )
    
    # Create memory
    memory = create_memory(
        model_name=model, 
        openai_api_key=openai_api_key or (api_key if provider.lower() == "openai" else None)
    )
    
    # Create prompts
    answer_prompt = PromptTemplate(
        template=get_answer_template(language),
        input_variables=["chat_history", "context", "question"]
    )
    condense_prompt = get_condense_prompt()
    
    # Build chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=response_generation_llm,
        retriever=retriever,
        memory=memory,
        condense_question_llm=condense_question_llm,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        chain_type="stuff",
        verbose=False,
        return_source_documents=True
    )
    
    return chain, memory
