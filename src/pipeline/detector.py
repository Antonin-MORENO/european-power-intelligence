import pandas as pd
import numpy as np
from config import ZSCORE_THRESHOLD

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne zscore et anomaly au dataframe de prix.
    """
    df = df.copy()
    
    mean = df["price_eur_mwh"].mean()
    std = df["price_eur_mwh"].std()
    
    df["zscore"] = (df["price_eur_mwh"] - mean) / std
    df["anomaly"] = df["zscore"].abs() > ZSCORE_THRESHOLD
    df["anomaly_type"] = None
    df.loc[df["zscore"] > ZSCORE_THRESHOLD, "anomaly_type"] = "spike"
    df.loc[df["zscore"] < -ZSCORE_THRESHOLD, "anomaly_type"] = "drop"
    
    return df

def get_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retourne uniquement les lignes anormales avec contexte.
    """
    anomalies = df[df["anomaly"]].copy()
    anomalies["price_vs_mean"] = (
        anomalies["price_eur_mwh"] - df["price_eur_mwh"].mean()
    ).round(2)
    
    return anomalies