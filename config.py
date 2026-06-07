import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# SMARD API
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
SMARD_FILTER = 4169  # Day-Ahead prices Germany
SMARD_REGION = "DE"
SMARD_RESOLUTION = "hour"

# Paths
DATA_DIR = "data/raw"
REPORTS_DIR = "data/reports"
VECTOR_STORE_DIR = "data/vectorstore"

# Anomaly detection
ZSCORE_THRESHOLD = 2.5

# Backtest
INITIAL_CAPITAL = 10000