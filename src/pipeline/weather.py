import requests
import pandas as pd

def get_weather_for_timestamp(dt, lat=51.5, lon=10.0):
    """
    Récupère météo horaire pour une datetime donnée.
    Coordonnées par défaut : centre Allemagne.
    """
    date_str = dt.strftime("%Y-%m-%d")
    
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,windspeed_10m,shortwave_radiation"
        f"&timezone=Europe%2FBerlin"
    )
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    hourly = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "windspeed": data["hourly"]["windspeed_10m"],
        "solar_radiation": data["hourly"]["shortwave_radiation"]
    })
    
    # Trouver l'heure la plus proche
    target_hour = dt.replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    hourly["diff"] = (hourly["datetime"] - target_hour).abs()
    closest = hourly.loc[hourly["diff"].idxmin()]
    
    return {
        "temperature": round(closest["temperature"], 1),
        "windspeed": round(closest["windspeed"], 1),
        "solar_radiation": round(closest["solar_radiation"], 1)
    }