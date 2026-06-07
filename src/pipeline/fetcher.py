import requests
import pandas as pd
import time
from config import SMARD_BASE_URL, SMARD_FILTER, SMARD_REGION, SMARD_RESOLUTION

def get_available_timestamps():
    url = f"{SMARD_BASE_URL}/{SMARD_FILTER}/{SMARD_REGION}/index_{SMARD_RESOLUTION}.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()["timestamps"]

def get_timeseries(timestamp):
    url = f"{SMARD_BASE_URL}/{SMARD_FILTER}/{SMARD_REGION}/{SMARD_FILTER}_{SMARD_REGION}_{SMARD_RESOLUTION}_{timestamp}.json"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()["series"]
    return data

def fetch_prices(n_last=12):
    """
    Récupère les n_last dernières périodes disponibles.
    Chaque période = ~1 semaine de données horaires.
    n_last=12 ≈ 3 mois de données.
    """
    timestamps = get_available_timestamps()
    timestamps = timestamps[-n_last:]

    all_data = []
    for ts in timestamps:
        series = get_timeseries(ts)
        all_data.extend(series)
        time.sleep(0.2)  # respecter le serveur

    df = pd.DataFrame(all_data, columns=["timestamp_ms", "price_eur_mwh"])
    df = df.dropna()
    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Berlin")
    df = df.drop(columns=["timestamp_ms"])
    df = df.set_index("datetime")
    df = df.sort_index()

    return df