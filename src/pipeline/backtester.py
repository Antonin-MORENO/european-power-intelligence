import pandas as pd
import numpy as np
from config import INITIAL_CAPITAL

def run_backtest(df: pd.DataFrame) -> dict:
    """
    Backtest a mean-reversion strategy on hourly Day-Ahead electricity prices.

    Strategy logic:
        - Buy  (signal = +1) when price is below the historical mean → expect reversion upward
        - Sell (signal = -1) when price is above the historical mean → expect reversion downward

    PnL is computed in EUR/MWh terms (price difference), scaled from INITIAL_CAPITAL.

    Args:
        df: DataFrame with a 'price_eur_mwh' column indexed by datetime.

    Returns:
        Dictionary containing the enriched DataFrame and key performance metrics.
    """
    
    df = df.copy()
    
    # Compute the mean price over the entire period as the reversion target
    mean_price = df["price_eur_mwh"].mean()

    # Generate trading signals based on price vs mean
    df["signal"] = 0
    df.loc[df["price_eur_mwh"] < mean_price, "signal"] = 1
    df.loc[df["price_eur_mwh"] > mean_price, "signal"] = -1

    # Hourly PnL = previous signal * current price change (EUR/MWh)
    # shift(1) ensures we trade on the signal from the previous hour (no look-ahead bias)
    df["price_change"] = df["price_eur_mwh"].diff()
    df["trade_pnl"] = df["signal"].shift(1) * df["price_change"]
    df["trade_pnl"] = df["trade_pnl"].fillna(0)

    # Cumulative PnL starting from INITIAL_CAPITAL
    df["cumulative_pnl"] = INITIAL_CAPITAL + df["trade_pnl"].cumsum()

    # --- Performance Metrics ---
    total_return = (df["cumulative_pnl"].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    
    # Annualised Sharpe ratio (8760 = hours in a year)
    # Measures risk-adjusted return: higher is better
    std = df["trade_pnl"].std()
    sharpe = (df["trade_pnl"].mean() / std * np.sqrt(8760)) if std > 0 else 0.0

    # Maximum drawdown: largest peak-to-trough decline in portfolio valu
    rolling_max = df["cumulative_pnl"].cummax()
    drawdown = (df["cumulative_pnl"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Hit rate: percentage of trades that were profitable
    hit_rate = (df["trade_pnl"] > 0).sum() / len(df[df["trade_pnl"] != 0]) * 100

    return {
        "df": df,
        "total_return_pct": round(total_return, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "hit_rate_pct": round(hit_rate, 2),
        "final_capital": round(df["cumulative_pnl"].iloc[-1], 2)
    }