"""Retriever factory with different strategies."""
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_community.document_transformers import LongContextReorder
from langchain_text_splitters import CharacterTextSplitter
try:
    from langchain_cohere import CohereRerank
except ImportError:
    CohereRerank = None

def create_base_retriever(vectorstore: VectorStore, search_type: str = "similarity", k: int = 4, score_threshold: float = None) -> BaseRetriever:
    """Create a basic vectorstore backed retriever."""
    search_kwargs = {"k": k}
    if score_threshold is not None and search_type == "similarity_score_threshold":
        search_kwargs["score_threshold"] = score_threshold
        
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )

def create_compression_retriever(embeddings, base_retriever: BaseRetriever, chunk_size: int = 500, k: int = 16, similarity_threshold: float = None) -> BaseRetriever:
    """Create a contextual compression retriever pipeline."""
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0, separator=". ")
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, k=k, similarity_threshold=similarity_threshold)
    reordering = LongContextReorder()
    
    pipeline = DocumentCompressorPipeline(
        transformers=[splitter, redundant_filter, relevant_filter, reordering]
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

def create_cohere_retriever(base_retriever: BaseRetriever, cohere_api_key: str, cohere_model: str = "rerank-multilingual-v2.0", top_n: int = 10) -> BaseRetriever:
    """Create a Cohere reranker retriever."""
    if CohereRerank is None:
        raise ImportError("langchain-cohere is not installed. Please run `pip install langchain-cohere` in your terminal to use the Cohere reranker.")
        
    compressor = CohereRerank(
        cohere_api_key=cohere_api_key, 
        model=cohere_model, 
        top_n=top_n
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

def create_retriever(vector_store: VectorStore, embeddings, retriever_type: str, **kwargs) -> BaseRetriever:
    """Dispatcher to create a retriever based on type."""
    base_k = kwargs.get("k", 4)
    base_retriever = create_base_retriever(vector_store, search_type=kwargs.get("search_type", "similarity"), k=kwargs.get("base_k", base_k * 4))
    
    if retriever_type == "Vectorstore backed retriever":
        return create_base_retriever(vector_store, search_type=kwargs.get("search_type", "similarity"), k=base_k)
    elif retriever_type == "Contextual compression":
        return create_compression_retriever(
            embeddings=embeddings,
            base_retriever=base_retriever,
            chunk_size=kwargs.get("chunk_size", 500),
            k=base_k,
            similarity_threshold=kwargs.get("similarity_threshold")
        )
    elif retriever_type == "Cohere reranker":
        cohere_api_key = kwargs.get("cohere_api_key")
        if not cohere_api_key:
            raise ValueError("cohere_api_key is required for Cohere reranker")
        return create_cohere_retriever(
            base_retriever=base_retriever,
            cohere_api_key=cohere_api_key,
            top_n=base_k
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
