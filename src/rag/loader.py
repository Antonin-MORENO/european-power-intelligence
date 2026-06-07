from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from config import REPORTS_DIR

def load_documents():
    """
    Load all PDF reports from the reports directory as LangChain Document objects.
    Each page is tagged with its source filename for citation in RAG responses.
    """
    docs = []
    for filename in os.listdir(REPORTS_DIR):
        if filename.endswith(".pdf"):
            path = os.path.join(REPORTS_DIR, filename)
            loader = PyPDFLoader(path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = filename
            docs.extend(pages)
            print(f"Loaded {len(pages)} pages from {filename}")
    return docs

def split_documents(docs):
    """
    Split documents into overlapping chunks for embedding and retrieval.

    chunk_size=800   : Balances context richness vs retrieval precision.
    chunk_overlap=100: Ensures no information is lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)