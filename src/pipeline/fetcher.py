import requests
import pandas as pd
import time
from config import SMARD_BASE_URL, SMARD_FILTER, SMARD_REGION, SMARD_RESOLUTION

def get_available_timestamps():
    """
    Fetch the list of available data timestamps from the SMARD API.

    SMARD organises its time series into weekly chunks, each identified by a
    Unix timestamp (ms). This endpoint returns all available chunk timestamps
    for the configured filter, region and resolution.

    Returns:
        List of integer timestamps (milliseconds) representing available data chunks.
    """
    url = f"{SMARD_BASE_URL}/{SMARD_FILTER}/{SMARD_REGION}/index_{SMARD_RESOLUTION}.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()["timestamps"]

def get_timeseries(timestamp):
    """
    Fetch the hourly price series for a single SMARD data chunk.

    Each chunk covers approximately one week of hourly data points.
    The response is a list of [timestamp_ms, price_eur_mwh] pairs.

    Args:
        timestamp: Unix timestamp (ms) identifying the data chunk to retrieve.

    Returns:
        List of [timestamp_ms, price_eur_mwh] pairs for the requested chunk.
    """
    url = f"{SMARD_BASE_URL}/{SMARD_FILTER}/{SMARD_REGION}/{SMARD_FILTER}_{SMARD_REGION}_{SMARD_RESOLUTION}_{timestamp}.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()["series"]
    return data

def fetch_prices(n_last=12):
    """
    Fetch and assemble the most recent hourly Day-Ahead prices from SMARD.

    Retrieves the n_last most recent weekly chunks and concatenates them into
    a single time-indexed DataFrame. A short sleep between requests avoids
    overwhelming the SMARD server.

    Default: n_last=12 ≈ 3 months of hourly data (~2,016 rows).

    Args:
        n_last: Number of weekly chunks to retrieve (1 chunk ≈ 1 week).

    Returns:
        DataFrame with a timezone-aware datetime index (Europe/Berlin) and
        a single column 'price_eur_mwh' containing hourly Day-Ahead prices.
    """
    
    # Retrieve all available chunk timestamps and keep only the most recent ones
    timestamps = get_available_timestamps()
    timestamps = timestamps[-n_last:]

    all_data = []
    for ts in timestamps:
        series = get_timeseries(ts)
        all_data.extend(series)
        time.sleep(0.2) 

    df = pd.DataFrame(all_data, columns=["timestamp_ms", "price_eur_mwh"])
    df = df.dropna()
    
        # Convert Unix ms timestamps to timezone-aware datetime index (Europe/Berlin)
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Berlin")
    df = df.drop(columns=["timestamp_ms"])
    df = df.set_index("datetime")
    df = df.sort_index()

    return df