import os
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.data_prep import load_and_prepare, make_lag_features
from utils.models import fit_sarimax_small_grid, fit_gbr_lag_model, ensemble_forecast
from utils.evaluate import backtest_multi_step
from utils.plotting import plot_history_forecast

DATAHUB_URL = "https://datahub.io/core/gold-prices/r/monthly.csv"

st.set_page_config(page_title="Gold Price Forecast", layout="wide")
st.title("📈 Gold Price Forecast (100y history → 25y outlook)")

with st.expander("About this app", expanded=False):
    st.markdown(
        """
        This dashboard fits two complementary models to your monthly gold price series:
        1) a **SARIMAX** (small grid search) and
        2) a **Gradient Boosting Regressor** on lag features,
        then ensembles them for stability on long horizons.

        **Data source helper:** If you don't have a file, the app can **auto-fetch** monthly gold prices
        (USD per troy ounce) from DataHub (1833–present) and optionally trim to **last 100 years**.
        """
    )

# --- Data input
st.sidebar.header("Data Input")
default_path = "data/gold_prices.csv"
src = st.sidebar.radio("Source", ["Upload CSV", f"Local file: {default_path}", "Auto-fetch (DataHub 1833–present)"])
trim_100y = st.sidebar.checkbox("Trim to last 100 years", value=True)
unit = st.sidebar.selectbox("Display/forecast unit", ["troy_ounce", "gram", "kilogram"], index=0)

df = None
if src == "Upload CSV":
    up = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
elif src.startswith("Local file"):
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
    else:
        st.warning(f"No local file at {default_path}. Choose Upload or Auto-fetch.")
else:
    # Auto-fetch
    try:
        df = pd.read_csv(DATAHUB_URL)
        # Normalize columns to expected names
        # DataHub has 'Date','Price'
        if {"Date","Price"}.issubset(df.columns):
            df = df.rename(columns={"Date":"date","Price":"price_usd"})
        # Some mirrors use lowercase already
        if {"date","price_usd"}.issubset(df.columns) is False:
            # Try alternative: 'value' column
            if "value" in df.columns and "date" in df.columns:
                df = df.rename(columns={"value":"price_usd"})
    except Exception as e:
        st.error(f"Auto-fetch failed: {e}")
        df = None

if df is None:
    st.info("Waiting for data… Upload a CSV, point to a local file, or use Auto-fetch.")
    st.stop()

# Trim to last 100y if selected
if trim_100y:
    try:
        dcol = "date" if "date" in df.columns else "Date"
        df[dcol] = pd.to_datetime(df[dcol])
        cutoff = pd.Timestamp.today() - pd.DateOffset(years=100)
        df = df[df[dcol] >= cutoff].copy()
    except Exception:
        pass

# Convert units if requested (dataset is USD per troy ounce by default)
if unit != "troy_ounce":
    # We'll convert after prep (on the monthly series)
    pass

with st.spinner("Preparing data…"):
    try:
        ts, meta = load_and_prepare(df, date_col=("date" if "date" in df.columns else "Date"), price_col=("price_usd" if "price_usd" in df.columns else "Price"), monthly=True, log_transform=True)
    except Exception as e:
        st.error(f"Data prep failed: {e}")
        st.stop()

# Apply unit conversion
if unit == "gram":
    ts["price_usd"] = ts["price_usd"] / 31.1035
    if "price_usd_log" in ts:
        ts["price_usd_log"] = np.log(ts["price_usd"])
elif unit == "kilogram":
    ts["price_usd"] = ts["price_usd"] * (1000.0 / 31.1035)
    if "price_usd_log" in ts:
        ts["price_usd_log"] = np.log(ts["price_usd"])

st.success(f"Loaded {len(ts)} monthly points from {ts.index.min().date()} → {ts.index.max().date()}")
st.caption(f"Unit: USD per {unit.replace('_',' ')}")
st.write(meta)

st.subheader("Historical Series")
st.line_chart(ts["price_usd"], height=250)

# --- Modeling controls
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
        # SARIMAX
        sarimax_res = fit_sarimax_small_grid(
            ts["price_usd_log" if use_log else "price_usd"],
            seasonal=seasonal
        )
        # GBR on lags
        feat_df = make_lag_features(ts["price_usd"], lags=lags)
        gbr_res = fit_gbr_lag_model(feat_df, target_col="price_usd", horizon=h)

        # Ensemble Forecast
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
            file_name=f"gold_forecast_{unit}.csv",
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
