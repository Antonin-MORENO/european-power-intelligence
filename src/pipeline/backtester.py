import pandas as pd
import numpy as np
from config import INITIAL_CAPITAL

def run_backtest(df: pd.DataFrame) -> dict:
    df = df.copy()
    mean_price = df["price_eur_mwh"].mean()

    # Signal : acheter sous la moyenne, vendre au dessus
    df["signal"] = 0
    df.loc[df["price_eur_mwh"] < mean_price, "signal"] = 1
    df.loc[df["price_eur_mwh"] > mean_price, "signal"] = -1

    # PnL par heure = signal * variation de prix (en €/MWh)
    df["price_change"] = df["price_eur_mwh"].diff()
    df["trade_pnl"] = df["signal"].shift(1) * df["price_change"]
    df["trade_pnl"] = df["trade_pnl"].fillna(0)

    # PnL cumulé
    df["cumulative_pnl"] = INITIAL_CAPITAL + df["trade_pnl"].cumsum()

    # Métriques
    total_return = (df["cumulative_pnl"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    std = df["trade_pnl"].std()
    sharpe = (df["trade_pnl"].mean() / std * np.sqrt(8760)) if std > 0 else 0.0

    rolling_max = df["cumulative_pnl"].cummax()
    drawdown = (df["cumulative_pnl"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    hit_rate = (df["trade_pnl"] > 0).sum() / len(df[df["trade_pnl"] != 0]) * 100

    return {
        "df": df,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "hit_rate_pct": round(hit_rate, 2),
        "final_capital": round(df["cumulative_pnl"].iloc[-1], 2)
    }