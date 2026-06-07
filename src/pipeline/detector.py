import pandas as pd
import numpy as np
from config import ZSCORE_THRESHOLD

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect abnormal price events in the Day-Ahead price series using z-score.

    A price is flagged as an anomaly if its z-score exceeds ZSCORE_THRESHOLD in
    either direction:
        - 'spike': price significantly above the mean (e.g. cold snap, low wind)
        - 'drop' : price significantly below the mean (e.g. solar surplus, low demand)

    Args:
        df: DataFrame with a 'price_eur_mwh' column indexed by datetime.

    Returns:
        Original DataFrame enriched with 'zscore', 'anomaly', and 'anomaly_type' columns.
    """
    df = df.copy()
    
    # Compute period mean and standard deviation as the baseline distribution
    mean = df["price_eur_mwh"].mean()
    std = df["price_eur_mwh"].std()
    
    # Z-score: how many standard deviations away from the mean each price is
    df["zscore"] = (df["price_eur_mwh"] - mean) / std
    
    # Flag as anomaly if absolute z-score exceeds the configured threshold
    df["anomaly"] = df["zscore"].abs() > ZSCORE_THRESHOLD
    
    # Classify anomaly direction
    df["anomaly_type"] = None
    df.loc[df["zscore"] > ZSCORE_THRESHOLD, "anomaly_type"] = "spike"
    df.loc[df["zscore"] < -ZSCORE_THRESHOLD, "anomaly_type"] = "drop"
    
    return df

def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and enrich anomalous rows for downstream analysis and LLM explanation.

    Adds 'price_vs_mean' to quantify how far each anomaly deviates from the
    period average in absolute EUR/MWh terms.

    Args:
        df: DataFrame output from detect_anomalies(), must contain 'anomaly' column.

    Returns:
        Filtered DataFrame containing only anomalous rows with deviation context.
    """
    anomalies = df[df["anomaly"]].copy()
    
    # Absolute deviation from mean: provides intuitive context for LLM prompts
    anomalies["price_vs_mean"] = (
        anomalies["price_eur_mwh"] - df["price_eur_mwh"].mean()
    ).round(2)
    
    return anomalies