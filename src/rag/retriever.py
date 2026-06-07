from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from src.rag.indexer import get_or_build_index
from config import GROQ_API_KEY, GROQ_MODEL

def retrieve(query: str, k: int = 4):
    """
    Retrieve the k most semantically similar document chunks for a given query.

    Args:
        query: Natural language question to search against the vector index.
        k    : Number of chunks to retrieve (default: 4).

    Returns:
        List of LangChain Document objects ranked by similarity.
    """
    vectorstore = get_or_build_index()
    docs = vectorstore.similarity_search(query, k=k)
    return docs

def answer_with_rag(query: str) -> dict:
    """
    Answer a question using Retrieval-Augmented Generation (RAG).

    Retrieves the most relevant chunks from the FAISS index, injects them as
    context into an LLM prompt, and returns the grounded answer with source citations.
    The LLM is instructed to answer solely from the provided context and explicitly
    acknowledge when the answer cannot be found — preventing hallucination.

    Args:
        query: Natural language question about European energy markets.

    Returns:
        Dict with 'answer' (str) and 'sources' (list of dicts with source, page, content).
    """
    docs = retrieve(query)
    
    
    # Format retrieved chunks with source metadata for transparent citation
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    ])
    
    prompt = f"""You are an expert European energy market analyst.
Answer the question below using ONLY the provided context from official energy reports.
Always cite the source document and page number.
If the answer is not in the context, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "answer": response.content,
        "sources": [
            {
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "?"),
                "content": doc.page_content[:200]
            }
            for doc in docs
        ]
    }