"""Main Streamlit application for the RAG Chatbot. (Updated)"""

import os
import streamlit as st

# Configure the Streamlit page before importing other modules to avoid errors
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

import sys
from pathlib import Path
# Add the project root directory to Python's path so it can find the 'src' module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TMP_DIR,
    VECTOR_STORE_DIR,
    LLM_PROVIDERS,
    OPENAI_MODELS,
    GOOGLE_MODELS,
    HUGGINGFACE_MODELS,
    GROQ_MODELS,
    RETRIEVER_TYPES,
    WELCOME_MESSAGES
)
from src.document_loader import load_documents, save_uploaded_files, delete_temp_files
from src.text_splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, load_vector_store, list_vector_stores
from src.retriever import create_retriever
from src.chain import create_chain

def clear_chat_history():
    """Clear chat history and reset memory."""
    if "language" in st.session_state:
        language = st.session_state.language
        welcome_msg = WELCOME_MESSAGES.get(language, WELCOME_MESSAGES["english"])
    else:
        welcome_msg = WELCOME_MESSAGES["english"]

    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]
    if "memory" in st.session_state:
        st.session_state.memory.clear()

def process_documents(uploaded_files, vectorstore_name, provider, api_key, model, temperature, top_p, retriever_type, cohere_api_key):
    """Process uploaded documents and create a vector store and chat chain."""
    if not api_key:
        st.warning("Please enter your API Key in the sidebar to proceed.")
        return
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return
    if not vectorstore_name:
        st.warning("Please provide a name for the vectorstore.")
        return

    try:
        with st.spinner("Creating vectorstore... This may take a moment."):
            # Clean up temp directory
            delete_temp_files(TMP_DIR)
            
            # Save uploaded files temporarily
            save_uploaded_files(uploaded_files, TMP_DIR)
            
            # Load documents
            documents = load_documents(TMP_DIR)
            if not documents:
                st.error("No content could be extracted from the uploaded files.")
                return
                
            # Split documents
            chunks = split_documents(documents)
            
            # Get embeddings
            embeddings = get_embeddings(provider, api_key)
            
            # Create vector store
            vector_store = create_vector_store(chunks, embeddings, vectorstore_name, VECTOR_STORE_DIR)
            
            # Create retriever
            retriever = create_retriever(vector_store, embeddings, retriever_type, cohere_api_key=cohere_api_key)
            
            # Create chain
            chain, memory = create_chain(
                retriever=retriever,
                provider=provider,
                model=model,
                api_key=api_key,
                language=st.session_state.language,
                temperature=temperature,
                top_p=top_p
            )
            
            # Store in session state
            st.session_state.chain = chain
            st.session_state.memory = memory
            st.session_state.retriever = retriever
            st.session_state.vectorstore_name = vectorstore_name
            
            clear_chat_history()
            st.success(f"Successfully processed {len(uploaded_files)} files and created vectorstore '{vectorstore_name}'!")
    except Exception as e:
        st.error(f"An error occurred during document processing: {str(e)}")

def load_existing_vectorstore(vectorstore_name, provider, api_key, model, temperature, top_p, retriever_type, cohere_api_key):
    """Load an existing vector store and create a chat chain."""
    if not api_key:
        st.warning("Please enter your API Key in the sidebar to proceed.")
        return
    if not vectorstore_name:
        st.warning("Please select a vectorstore to load.")
        return

    try:
        with st.spinner(f"Loading vectorstore '{vectorstore_name}'..."):
            # Get embeddings
            embeddings = get_embeddings(provider, api_key)
            
            # Load vector store
            vector_store = load_vector_store(vectorstore_name, embeddings, VECTOR_STORE_DIR)
            
            if not vector_store:
                st.error(f"Could not load vectorstore '{vectorstore_name}'.")
                return
                
            # Create retriever
            retriever = create_retriever(vector_store, embeddings, retriever_type, cohere_api_key=cohere_api_key)
            
            # Create chain
            chain, memory = create_chain(
                retriever=retriever,
                provider=provider,
                model=model,
                api_key=api_key,
                language=st.session_state.language,
                temperature=temperature,
                top_p=top_p
            )
            
            # Store in session state
            st.session_state.chain = chain
            st.session_state.memory = memory
            st.session_state.retriever = retriever
            st.session_state.vectorstore_name = vectorstore_name
            
            clear_chat_history()
            st.success(f"Successfully loaded vectorstore '{vectorstore_name}'!")
    except Exception as e:
        st.error(f"An error occurred while loading the vectorstore: {str(e)}")

def chatbot():
    """Main application layout and logic."""
    # Initialize session state for language if not exists
    if "language" not in st.session_state:
        st.session_state.language = "english"

    # Sidebar setup
    with st.sidebar:
        st.title("🤖 RAG Chatbot Settings")
        st.caption("Configure your AI assistant")
        
        # Provider selection
        provider = st.radio("LLM Provider", LLM_PROVIDERS)
        
        # API Key input based on provider
        api_key_label = f"{provider} API Key"
        if provider == "Google":
            api_key_label = "Google Gemini API Key"
        api_key = st.text_input(api_key_label, type="password")
        
        # Models and parameters
        with st.expander("Models and parameters", expanded=True):
            if provider == "OpenAI":
                models = OPENAI_MODELS
            elif provider == "Google":
                models = GOOGLE_MODELS
            elif provider == "HuggingFace":
                models = HUGGINGFACE_MODELS
            elif provider == "Groq":
                models = GROQ_MODELS
            else:
                models = ["default"]
                
            model = st.selectbox("Model", models)
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
            top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.05)
            
        # Language selection
        languages = list(WELCOME_MESSAGES.keys())
        language = st.selectbox("Chatbot Language", languages, index=languages.index(st.session_state.language))
        if language != st.session_state.language:
            st.session_state.language = language
            # Update welcome message if chat history only has the old welcome message
            if len(st.session_state.get("messages", [])) <= 1:
                clear_chat_history()
                
        st.divider()
        
        # Retriever settings
        st.subheader("Retrieval Settings")
        retriever_type = st.selectbox("Retriever Type", RETRIEVER_TYPES)
        
        cohere_api_key = ""
        if "Cohere" in retriever_type:
            cohere_api_key = st.text_input("Cohere API Key (for reranking)", type="password")
            
        st.info("Settings changes take effect when you create or load a vectorstore.")

    # Main Area
    st.header("📚 Document Knowledge Base")
    
    tab1, tab2 = st.tabs(["Create New Vectorstore", "Open Saved Vectorstore"])
    
    with tab1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files (PDF, TXT, DOCX, CSV)", 
            type=["pdf", "txt", "docx", "csv"],
            accept_multiple_files=True
        )
        new_vs_name = st.text_input("New Vectorstore Name", placeholder="e.g., my_knowledge_base")
        
        if st.button("Create Vectorstore", key="create_vs_btn"):
            process_documents(
                uploaded_files=uploaded_files,
                vectorstore_name=new_vs_name,
                provider=provider,
                api_key=api_key,
                model=model,
                temperature=temperature,
                top_p=top_p,
                retriever_type=retriever_type,
                cohere_api_key=cohere_api_key
            )
            
    with tab2:
        st.subheader("Load Existing Vectorstore")
        available_stores = list_vector_stores(VECTOR_STORE_DIR)
        
        if not available_stores:
            st.info("No saved vectorstores found. Create one in the other tab.")
        else:
            selected_vs_name = st.selectbox("Select Vectorstore", available_stores)
            
            if st.button("Load Vectorstore", key="load_vs_btn"):
                load_existing_vectorstore(
                    vectorstore_name=selected_vs_name,
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    retriever_type=retriever_type,
                    cohere_api_key=cohere_api_key
                )
                
    st.divider()
    
    # Chat Interface
    col1, col2 = st.columns([4, 1])
    with col1:
        vs_status = f" (Using: {st.session_state.vectorstore_name})" if "vectorstore_name" in st.session_state else ""
        st.header(f"💬 Chat with your data{vs_status}")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history, use_container_width=True)
        
    # Initialize messages
    if "messages" not in st.session_state:
        clear_chat_history()
        
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "source_documents" in msg:
                with st.expander("Source documents"):
                    for i, doc in enumerate(msg["source_documents"]):
                        source = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**Source {i+1}:** {os.path.basename(source)}")
                        st.markdown(f"```{doc.page_content}```")
                        
    # Chat Input
    if prompt := st.chat_input("Ask a question about your documents..."):
        if "chain" not in st.session_state:
            st.warning("Please create or load a vectorstore first.")
            return
            
        if not api_key:
            st.warning("Please enter your API Key in the sidebar.")
            return
            
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke({"question": prompt})
                    
                    # Process response based on provider/chain output type
                    answer = response.get("answer", "")
                    
                    # Handle HuggingFace answer format if it includes the prompt
                    if provider == "HuggingFace" and "Helpful Answer:" in answer:
                        answer = answer.split("Helpful Answer:")[-1].strip()
                        
                    source_docs = response.get("source_documents", [])
                    
                    st.markdown(answer)
                    
                    if source_docs:
                        with st.expander("Source documents"):
                            for i, doc in enumerate(source_docs):
                                source = doc.metadata.get("source", "Unknown")
                                st.markdown(f"**Source {i+1}:** {os.path.basename(source)}")
                                st.markdown(f"```{doc.page_content}```")
                                
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "source_documents": source_docs
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    chatbot()
