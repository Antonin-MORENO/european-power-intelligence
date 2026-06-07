from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL
import pandas as pd

llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

def explain_anomaly(row: pd.Series, mean_price: float) -> str:
    """
    Envoie une anomalie au LLM et retourne une explication courte.
    """
    prompt = f"""You are a European electricity market analyst.
    
An unusual price event was detected on the German Day-Ahead power market:
- Datetime: {row.name}
- Price: {row['price_eur_mwh']:.2f} EUR/MWh
- Average price over period: {mean_price:.2f} EUR/MWh
- Deviation: {row['price_vs_mean']:.2f} EUR/MWh ({row['anomaly_type']})
- Z-score: {row['zscore']:.2f}

In 2-3 sentences, explain the most likely fundamental reasons for this price {row['anomaly_type']} 
(consider: wind/solar generation, demand, gas prices, grid constraints, seasonality).
Be specific and concise."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def explain_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne explanation pour chaque anomalie.
    """
    anomalies = df[df["anomaly"]].copy()
    mean_price = df["price_eur_mwh"].mean()
    
    anomalies["explanation"] = anomalies.apply(
        lambda row: explain_anomaly(row, mean_price), axis=1
    )
    
    return anomalies