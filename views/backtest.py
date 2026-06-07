import streamlit as st
import plotly.graph_objects as go
from src.pipeline.fetcher import fetch_prices
from src.pipeline.detector import detect_anomalies
from src.pipeline.backtester import run_backtest

st.title("📈 Backtest Explorer")
st.caption("Mean-reversion strategy on German Day-Ahead prices")

# --- Sidebar: user-configurable parameters ---
with st.sidebar:
    st.header("Parameters")
    # Number of weekly chunks to fetch — more weeks = longer backtest horizon
    n_weeks = st.slider("Weeks of data", min_value=4, max_value=52, value=12)
    run = st.button("Run Backtest", type="primary")

if run:
    # Fetch and prepare price data with anomaly flags
    with st.spinner("Fetching data..."):
        df = fetch_prices(n_last=n_weeks)
        df = detect_anomalies(df)

    result = run_backtest(df)
    df_bt = result["df"]

    # --- Performance metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Capital", f"€{result['final_capital']:,.0f}", f"{result['total_return_pct']}%")
    col2.metric("Sharpe Ratio", result['sharpe_ratio'])       # Risk-adjusted return
    col3.metric("Max Drawdown", f"{result['max_drawdown_pct']}%")  # Worst peak-to-trough loss
    col4.metric("Hit Rate", f"{result['hit_rate_pct']}%")     # % of profitable trades

    # --- Chart 1: Cumulative PnL over time ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["cumulative_pnl"],
        mode="lines", fill="tozeroy",
        name="Portfolio Value",
        line=dict(color="#00C853", width=1.5)
    ))
    fig1.update_layout(
        title="Cumulative PnL",
        xaxis_title="Date",
        yaxis_title="EUR",
        height=350
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Chart 2: Price series with buy/sell signals overlaid ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_bt.index, y=df_bt["price_eur_mwh"],
        mode="lines", name="Price",
        line=dict(color="#4C9BE8", width=1)
    ))

    # Buy signals: price below mean, expecting upward reversion
    buys = df_bt[df_bt["signal"] == 1]
    # Sell signals: price above mean, expecting downward reversion
    sells = df_bt[df_bt["signal"] == -1]

    fig2.add_trace(go.Scatter(
        x=buys.index, y=buys["price_eur_mwh"],
        mode="markers", name="Buy",
        marker=dict(color="green", size=4, symbol="triangle-up")
    ))
    fig2.add_trace(go.Scatter(
        x=sells.index, y=sells["price_eur_mwh"],
        mode="markers", name="Sell",
        marker=dict(color="red", size=4, symbol="triangle-down")
    ))
    fig2.update_layout(
        title="Price & Signals",
        xaxis_title="Date",
        yaxis_title="EUR/MWh",
        height=350
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Set parameters and click **Run Backtest** to start.")