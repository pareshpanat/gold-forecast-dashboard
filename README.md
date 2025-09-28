# Gold Price Forecast Dashboard (Python + Streamlit)

Forecast long-horizon gold prices using classical time-series (SARIMAX) and ML (Gradient Boosting on lagged features), then ensemble the forecasts. Designed for **100+ years** of monthly data and **up to 25 years** horizon.

## Quick Start

```bash
# 1) Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2) Install deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3) Prepare your data
# Put your CSV at data/gold_prices.csv with columns: date,price_usd
# - date: YYYY-MM-DD (month-end is fine)
# - price_usd: numeric

# 4) Launch the dashboard
streamlit run app/dashboard.py
```

## CSV Format

Minimal required columns:
- `date` (YYYY-MM-DD)
- `price_usd` (float)

Optional columns (if you have them): `cpi`, `usd_index`, `oil_price`, etc.
The app will ignore unknown columns unless you select them as regressors.

### Example (head)

```csv
date,price_usd
1925-01-31,20.67
1925-02-28,20.67
...
```

## What’s inside?

- `app/dashboard.py` — Streamlit UI; upload/select data, train models, see forecasts, download CSVs.
- `app/utils/data_prep.py` — loading, cleaning, resampling to monthly, feature engineering (lags, returns).
- `app/utils/models.py` — SARIMAX (small grid search), GradientBoostingRegressor (lags), ensembling, forecasting.
- `app/utils/evaluate.py` — rolling-origin backtests (TimeSeriesSplit), metrics.
- `app/utils/plotting.py` — Plotly charts.
- `notebooks/` — space for your experimentation (empty by default).

## Modeling Approach

1. **Data handling**
   - Ensure monthly frequency (`MS` or `M`), forward-fill small gaps.
   - Optionally log-transform price to model growth/stationarity.

2. **Models**
   - **SARIMAX** with a tiny grid over `(p,d,q)×(P,D,Q,12)` (seasonality 12 months).
   - **Gradient Boosting** with lagged features (1, 3, 6, 12, 24, 60) and rolling stats.
   - **Ensemble**: average (or weighted) of the two to stabilize long-horizon drift.

3. **Validation**
   - **Rolling TimeSeriesSplit** with multi-step predictions.
   - Report MAPE, RMSE.

4. **Forecast**
   - Up to 300 months (25 years). Outputs both **level** and **log-level** forecasts.

## Notes & Caveats

- Long-horizon (25y) forecasts are **highly uncertain**. Treat outputs as exploratory scenarios, not investment advice.
- If you have exogenous drivers (e.g., CPI, USD index), you can add them as regressors.
- For robust research, include structural breaks and try multiple horizons & windows; consider probabilistic intervals.

## License
MIT
