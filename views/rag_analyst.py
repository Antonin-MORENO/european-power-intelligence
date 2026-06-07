import streamlit as st
from src.rag.retriever import answer_with_rag

st.title("🧠 RAG Analyst")
st.caption("Ask questions about European energy markets — answers grounded in official reports")

# --- User input ---
query = st.text_input(
    "Your question",
    placeholder="What are the main risks for European power prices in winter 2025?"
)
run = st.button("Ask", type="primary")

if run and query:
    # Retrieve relevant chunks from FAISS index and generate grounded LLM answer
    with st.spinner("Searching reports and generating answer..."):
        result = answer_with_rag(query)

    # Display the LLM-generated answer
    st.subheader("Answer")
    st.write(result["answer"])

    # Display source chunks used to ground the answer — ensures full transparency
    st.subheader("Sources")
    for s in result["sources"]:
        with st.expander(f"{s['source']} — page {s['page']}"):
            st.caption(s["content"])

elif run and not query:
    st.warning("Please enter a question.")

else:
    st.info("Enter a question and click **Ask** to query the reports.")