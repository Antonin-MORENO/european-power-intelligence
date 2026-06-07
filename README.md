# ⚡ European Power Intelligence

A live market intelligence dashboard for the German Day-Ahead electricity market, combining automated anomaly detection, strategy backtesting, and a RAG-powered research assistant grounded in official energy reports.

🔗 **[Live Dashboard](https://european-power-intelligence-antonin-alex-moreno.streamlit.app/)**

---

## Overview

This project demonstrates an end-to-end data science workflow applied to European energy markets — from raw market data ingestion to LLM-powered analysis — directly aligned with quantitative trading and data science workflows used in energy trading firms.

The dashboard covers three core capabilities:

- **Anomaly Pipeline** — automated detection of unusual price events on live SMARD data, enriched with real meteorological conditions and explained by an LLM
- **Backtest Explorer** — prototype and evaluate a mean-reversion trading strategy with standard financial metrics (Sharpe ratio, max drawdown, hit rate, PnL)
- **RAG Analyst** — question-answering system grounded in official IEA, ENTSO-E and Ember reports, with transparent source citations

---

## Architecture

```
european-power-intelligence/
├── app.py                    # Streamlit entry point & navigation
├── config.py                 # Centralised constants and API settings
├── data/
│   └── vectorstore/          # Pre-built FAISS index (included in repo)
├── src/
│   ├── pipeline/
│   │   ├── fetcher.py        # SMARD API client — hourly Day-Ahead prices
│   │   ├── detector.py       # Z-score based anomaly detection
│   │   ├── explainer.py      # LLM explanation enriched with weather data
│   │   ├── weather.py        # Open-Meteo API client — wind, temp, solar
│   │   └── backtester.py     # Mean-reversion strategy & performance metrics
│   └── rag/
│       ├── loader.py         # PDF ingestion and chunking
│       ├── indexer.py        # FAISS vector index builder
│       └── retriever.py      # Semantic search + LLM answer generation
└── views/
    ├── anomaly_pipeline.py   # Streamlit page 1
    ├── backtest.py           # Streamlit page 2
    └── rag_analyst.py        # Streamlit page 3
```

---

## Features

### 🔍 Anomaly Pipeline

- Fetches live hourly Day-Ahead prices from the **SMARD API** (Bundesnetzagentur) — no API key required
- Detects price spikes and drops using **z-score statistics**, with a user-adjustable sensitivity threshold
- Enriches each anomaly with real weather data from **Open-Meteo** (wind speed, temperature, solar radiation) for the exact timestamp and location
- Sends enriched context to **Llama 3.1 via Groq API** to generate a 2-3 sentence fundamental explanation linking meteorological conditions to grid supply/demand imbalances

### 📈 Backtest Explorer

- Backtests a **mean-reversion strategy** on historical Day-Ahead prices
- Signal logic: buy when price is below the period mean (expect upward reversion), sell when above
- Uses `shift(1)` to eliminate look-ahead bias — signals are based strictly on prior-hour information
- Computes standard quantitative trading metrics:
  - **Sharpe Ratio** — annualised risk-adjusted return (√8760 for hourly data)
  - **Maximum Drawdown** — largest peak-to-trough portfolio decline
  - **Hit Rate** — percentage of profitable trades
  - **Cumulative PnL** — portfolio value over time from a €10,000 initial capital

### 🧠 RAG Analyst

- Ingests 5 official energy reports (562 pages, 1,692 chunks) into a **FAISS vector index**
- Embeddings generated with `all-MiniLM-L6-v2` via **HuggingFace Sentence Transformers**
- Retrieves the most semantically relevant chunks via **cosine similarity search**
- Passes retrieved context to **Llama 3.1** with a strict grounding prompt — the LLM is instructed to answer solely from provided sources and acknowledge gaps explicitly, preventing hallucination
- Returns answers with full **source citations** (document name + page number)

**Knowledge base:**
- ENTSO-E Winter Outlook 2025-2026
- ENTSO-E Summer Outlook 2026
- Ember Global Electricity Review 2026
- IEA Electricity 2026
- IEA Gas Market Report Q1 & Q2 2026

> The FAISS vector index is pre-built and included in the repository (`data/vectorstore/`). No PDF downloads or index rebuilding required.

---

## Stack

| Layer | Technology |
|---|---|
| Data — Prices | SMARD API (Bundesnetzagentur) |
| Data — Weather | Open-Meteo Archive API |
| LLM | Llama 3.1 8B via Groq API |
| RAG Framework | LangChain + FAISS |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Dashboard | Streamlit |
| Visualisation | Plotly |

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/Antonin-MORENO/european-power-intelligence.git
cd european-power-intelligence
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure environment variables**
```bash
cp .env.example .env
# Add your Groq API key to .env
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

**5. Run the dashboard**
```bash
streamlit run app.py
```

---

## Data Sources

All data sources are **free and publicly accessible**:

- **SMARD** (smard.de) — German Day-Ahead electricity prices, updated daily by the Bundesnetzagentur. Accessed via public REST API, no authentication required.
- **Open-Meteo** (open-meteo.com) — Historical weather archive, no API key required.
- **ENTSO-E, IEA, Ember** — Official energy reports used to build the RAG knowledge base.

---

*Project created by Antonin MORENO - Open Source under MIT License.*