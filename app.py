import streamlit as st

# --- App-wide configuration ---
st.set_page_config(
    page_title="European Power Intelligence",
    page_icon="⚡",
    layout="wide"
)

# --- Navigation structure ---
# Pages are grouped into sections visible in the sidebar
pages = {
    "Market Analysis": [
        st.Page("views/anomaly_pipeline.py", title="Anomaly Pipeline", icon="🔍"),
        st.Page("views/backtest.py", title="Backtest Explorer", icon="📈"),
    ],
    "Research": [
        st.Page("views/rag_analyst.py", title="RAG Analyst", icon="🧠"),
    ]
}

pg = st.navigation(pages)
pg.run()