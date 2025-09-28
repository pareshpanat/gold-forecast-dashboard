import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_prep import load_and_prepare, make_lag_features
from utils.models import fit_sarimax_small_grid, fit_gbr_lag_model, ensemble_forecast
from utils.evaluate import backtest_multi_step
from utils.plotting import plot_history_forecast

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast (100y history → 25y outlook)")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        This dashboard fits two complementary models to your monthly gold price series:
        1) a **SARIMAX** (small grid search) and
        2) a **Gradient Boosting Regressor** on lag features,
        then ensembles them for stability on long horizons.

        **Tip:** Upload a CSV with at least `date,price_usd` columns. The more history the better.
        """
    )

st.sidebar.header("Data Input")
default_path = "data/gold_prices.csv"
src = st.sidebar.radio("Source", ["Upload CSV", f"Local file: {default_path}"])

df = None
if src == "Upload CSV":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
else:
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning(f"No local file at {default_path}. Upload a CSV instead.")

if df is None:
    st.info("Waiting for data… Upload a CSV (date,price_usd).")
    st.stop()

with st.spinner("Preparing data…"):
    try:
        ts, meta = load_and_prepare(df, date_col="date", price_col="price_usd", monthly=True, log_transform=True)
    except Exception as e:
        st.error(f"Data prep failed: {e}")
        st.stop()

st.success(f"Loaded {len(ts)} monthly points from {ts.index.min().date()} → {ts.index.max().date()}")
st.write(meta)

st.subheader("Historical Series")
st.line_chart(ts["price_usd"], height=250)

st.sidebar.header("Model Settings")
h_years = st.sidebar.slider("Forecast horizon (years)", 1, 25, 25)
h = h_years * 12
seasonal = st.sidebar.checkbox("Use seasonal SARIMAX (12)", value=True)
use_log = st.sidebar.checkbox("Model on log(price)", value=True)
lags = st.sidebar.multiselect("Lag features (months)", [1,3,6,12,24,60], default=[1,3,6,12,24,60])

st.sidebar.divider()
do_backtest = st.sidebar.checkbox("Run backtest (slow)", value=False)
folds = st.sidebar.slider("Backtest folds", 2, 6, 3)

if st.sidebar.button("Train & Forecast"):
    with st.spinner("Training models & forecasting…"):
        sarimax_res = fit_sarimax_small_grid(
            ts["price_usd_log" if use_log else "price_usd"],
            seasonal=seasonal
        )
        feat_df = make_lag_features(ts["price_usd"], lags=lags)
        gbr_res = fit_gbr_lag_model(feat_df, target_col="price_usd", horizon=h)

        fc_df = ensemble_forecast(
            ts=ts,
            sarimax_res=sarimax_res,
            gbr_res=gbr_res,
            horizon=h,
            use_log=use_log
        )

        st.subheader("Forecast")
        fig = plot_history_forecast(ts["price_usd"], fc_df["forecast"], fc_df.get("lower"), fc_df.get("upper"))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "⬇️ Download forecast CSV",
            data=fc_df.to_csv(index=True).encode("utf-8"),
            file_name="gold_forecast.csv",
            mime="text/csv",
        )

        if do_backtest:
            st.subheader("Backtest")
            bt = backtest_multi_step(
                series=ts["price_usd_log" if use_log else "price_usd"],
                build_fn=lambda s: fit_sarimax_small_grid(s, seasonal=seasonal),
                horizon=12,
                n_splits=folds
            )
            st.dataframe(bt)
else:
    st.info("Set options on the left, then click **Train & Forecast**.")
