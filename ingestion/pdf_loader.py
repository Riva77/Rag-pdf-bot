from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
import os

def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return a list of Document objects."""
    if not validate_pdf(file_path):
        raise ValueError(f"Invalid PDF file: {file_path}")
    loader = PyPDFLoader(file_path)
    return loader.load()

def validate_pdf(file_path: str) -> bool:
    """Check if PDF exists and has a valid extension."""
    return os.path.exists(file_path) and file_path.lower().endswith(".pdf")




