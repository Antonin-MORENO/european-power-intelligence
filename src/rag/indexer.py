from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from src.rag.loader import load_documents, split_documents
import os
from config import VECTOR_STORE_DIR

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_index():
    print("Loading documents...")
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_STORE_DIR)
    print("Index saved.")
    return vectorstore

def load_index():
    embeddings = get_embeddings()
    return FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

def get_or_build_index():
    if os.path.exists(VECTOR_STORE_DIR):
        return load_index()
    return build_index()