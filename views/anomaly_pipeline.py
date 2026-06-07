import streamlit as st
import plotly.graph_objects as go
from src.pipeline.fetcher import fetch_prices
from src.pipeline.detector import detect_anomalies, get_anomaly_summary
from src.pipeline.explainer import explain_all_anomalies

st.title("⚡ Anomaly Pipeline")
st.caption("Real-time detection of unusual price events on the German Day-Ahead market")

# Sidebar controls
with st.sidebar:
    st.header("Parameters")
    n_weeks = st.slider("Weeks of data", min_value=2, max_value=24, value=8)
    threshold = st.slider("Z-score threshold", min_value=1.5, max_value=4.0, value=2.5, step=0.1)
    run = st.button("Run Pipeline", type="primary")

if run:
    with st.spinner("Fetching prices from SMARD..."):
        df = fetch_prices(n_last=n_weeks)

    with st.spinner("Detecting anomalies..."):
        df = detect_anomalies(df)
        df["zscore_threshold"] = threshold
        df["anomaly"] = df["zscore"].abs() > threshold
        df["anomaly_type"] = None
        df.loc[df["zscore"] > threshold, "anomaly_type"] = "spike"
        df.loc[df["zscore"] < -threshold, "anomaly_type"] = "drop"
        anomalies = get_anomaly_summary(df)

    # Price chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["price_eur_mwh"],
        mode="lines", name="Price",
        line=dict(color="#4C9BE8", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies["price_eur_mwh"],
        mode="markers", name="Anomaly",
        marker=dict(color="red", size=8, symbol="x")
    ))
    fig.update_layout(
        title="Day-Ahead Prices with Anomalies",
        xaxis_title="Date",
        yaxis_title="EUR/MWh",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    # Métriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Total hours", len(df))
    col2.metric("Anomalies detected", len(anomalies))
    col3.metric("Anomaly rate", f"{len(anomalies)/len(df)*100:.1f}%")

    # Explications LLM
    if len(anomalies) > 0:
        with st.spinner("Generating LLM explanations..."):
            explained = explain_all_anomalies(anomalies)

        st.subheader("LLM Explanations")
        for _, row in explained.iterrows():
            with st.expander(f"{row.name.strftime('%Y-%m-%d %H:%M')} — {row['price_eur_mwh']:.1f} EUR/MWh ({row['anomaly_type']})"):
                st.write(row["explanation"])
else:
    st.info("Set parameters and click **Run Pipeline** to start.")