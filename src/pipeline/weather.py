import requests
import pandas as pd

def get_weather_for_timestamp(dt, lat=51.5, lon=10.0):
    """
    Fetch hourly weather conditions for a given datetime from Open-Meteo archive API.

    Retrieves temperature, wind speed and solar radiation for the hour closest to dt,
    used to ground LLM anomaly explanations in real meteorological conditions.
    Default coordinates (lat=51.5, lon=10.0) represent central Germany (DE-LU zone).

    Args:
        dt : Timezone-aware datetime of the price anomaly.
        lat: Latitude (default: central Germany).
        lon: Longitude (default: central Germany).

    Returns:
        Dict with 'temperature' (°C), 'windspeed' (km/h), 'solar_radiation' (W/m²).
    """
    date_str = dt.strftime("%Y-%m-%d")
    
    # Build Open-Meteo archive request for the full day containing the target datetime
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
    
    # Parse the hourly response into a DataFrame for easy time-based lookup
    hourly = pd.DataFrame({
        "datetime": pd.to_datetime(data["hourly"]["time"]),
        "temperature": data["hourly"]["temperature_2m"],
        "windspeed": data["hourly"]["windspeed_10m"],
        "solar_radiation": data["hourly"]["shortwave_radiation"]
    })
    
    # Strip timezone info from target for comparison (Open-Meteo returns naive datetimes)
    target_hour = dt.replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
    
    # Find the closest available hourly record to the requested timestamp
    hourly["diff"] = (hourly["datetime"] - target_hour).abs()
    closest = hourly.loc[hourly["diff"].idxmin()]
    
    return {
        "temperature": round(closest["temperature"], 1),
        "windspeed": round(closest["windspeed"], 1),
        "solar_radiation": round(closest["solar_radiation"], 1)
    }