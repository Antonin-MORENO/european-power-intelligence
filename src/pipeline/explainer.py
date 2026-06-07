from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL
import pandas as pd
from src.pipeline.weather import get_weather_for_timestamp

# Initialise the Groq LLM client once at module level to avoid repeated instantiation
llm = ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

def explain_anomaly(row: pd.Series, mean_price: float) -> str:
    """
    Generate a natural language explanation for a single price anomaly using an LLM.

    Enriches the prompt with real weather data (wind speed, temperature, solar radiation)
    fetched from Open-Meteo for the exact timestamp of the anomaly, allowing the LLM
    to ground its explanation in actual meteorological conditions rather than speculation.

    Args:
        row       : A single row from the anomaly DataFrame, indexed by datetime.
        mean_price: The period mean price in EUR/MWh, used as the baseline reference.

    Returns:
        A 2-3 sentence analytical explanation linking weather conditions to price behaviour.
    """
    # Fetch weather data for the anomaly timestamp
    # Gracefully degrade if the API call fails — explanation still runs without weather
    try:
        weather = get_weather_for_timestamp(row.name)
        weather_context = (
            f"- Wind speed: {weather['windspeed']} km/h\n"
            f"- Temperature: {weather['temperature']}°C\n"
            f"- Solar radiation: {weather['solar_radiation']} W/m²"
        )
    except:
        weather_context = "- Weather data unavailable"
        
    # Structured prompt providing full market and weather context to the LLM
    prompt = f"""You are a European electricity market analyst.

An unusual price event was detected on the German Day-Ahead power market:
- Datetime: {row.name}
- Price: {row['price_eur_mwh']:.2f} EUR/MWh
- Average price over period: {mean_price:.2f} EUR/MWh
- Deviation: {row['price_vs_mean']:.2f} EUR/MWh ({row['anomaly_type']})
- Z-score: {row['zscore']:.2f}

Weather conditions at that time:
{weather_context}

In 2-3 sentences, explain the most likely fundamental reasons for this price {row['anomaly_type']}
linking the weather data to grid supply/demand balance.
Be specific and concise."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def explain_all_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply LLM-based explanation to all detected anomalies in the DataFrame.

    Iterates over anomalous rows and calls explain_anomaly() for each one,
    adding the result as a new 'explanation' column for display in the dashboard.

    Args:
        df: Full price DataFrame containing an 'anomaly' boolean column.

    Returns:
        Filtered DataFrame of anomalous rows enriched with an 'explanation' column.
    """
    anomalies = df[df["anomaly"]].copy()
    mean_price = df["price_eur_mwh"].mean()
    
    anomalies["explanation"] = anomalies.apply(
        lambda row: explain_anomaly(row, mean_price), axis=1
    )
    
    return anomalies