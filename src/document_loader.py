"""Document loading module using LangChain loaders."""
import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    DirectoryLoader
)

def load_documents(directory: Path) -> List[Document]:
    """Load documents from a directory."""
    documents = []
    
    # PDF
    pdf_loader = DirectoryLoader(
        str(directory), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())
    
    # TXT
    txt_loader = DirectoryLoader(
        str(directory), glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())
    
    # CSV
    csv_loader = DirectoryLoader(
        str(directory), 
        glob="**/*.csv", 
        loader_cls=CSVLoader, 
        loader_kwargs={"encoding": "utf8"},
        show_progress=True
    )
    documents.extend(csv_loader.load())
    
    # DOCX
    docx_loader = DirectoryLoader(
        str(directory), glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True
    )
    documents.extend(docx_loader.load())
    
    return documents

def save_uploaded_files(uploaded_files: list, tmp_dir: Path) -> None:
    """Save uploaded files to a temporary directory."""
    for file in uploaded_files:
        file_path = tmp_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getvalue())

def delete_temp_files(tmp_dir: Path) -> None:
    """Delete all files in a temporary directory."""
    for file in tmp_dir.iterdir():
        if file.is_file():
            file.unlink()
