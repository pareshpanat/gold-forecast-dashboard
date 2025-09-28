
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from typing import Dict, Any
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

def _fit_one_sarimax(series, order, seasonal_order):
    mod = sm.tsa.statespace.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = mod.fit(disp=False)
    return res

def fit_sarimax_small_grid(series, seasonal=True) -> Dict[str, Any]:
    candidates = [
        ((0,1,1), (0,1,1,12) if seasonal else (0,0,0,0)),
        ((1,1,1), (0,1,1,12) if seasonal else (0,0,0,0)),
        ((2,1,1), (0,1,1,12) if seasonal else (0,0,0,0)),
        ((1,1,2), (1,1,0,12) if seasonal else (0,0,0,0)),
    ]
    best_res = None
    best_aic = np.inf
    for order, seas in candidates:
        try:
            res = _fit_one_sarimax(series, order, seas)
            if res.aic < best_aic:
                best_aic = res.aic
                best_res = res
                best = (order, seas)
        except Exception:
            continue
    if best_res is None:
        best_res = _fit_one_sarimax(series, (1,1,0), (0,0,0,0))
        best = ((1,1,0), (0,0,0,0))
    return {"model": best_res, "order": best[0], "seasonal_order": best[1], "aic": float(best_res.aic)}

def fit_gbr_lag_model(feat_df: pd.DataFrame, target_col="price_usd", horizon=12) -> Dict[str, Any]:
    X = feat_df.drop(columns=[target_col])
    y = feat_df[target_col].values

    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_rmse = np.inf

    for depth in [2,3]:
        for lr in [0.05, 0.1]:
            for n_est in [200, 400]:
                rmse_folds = []
                for train_idx, val_idx in tscv.split(X):
                    m = GradientBoostingRegressor(
                        max_depth=depth,
                        learning_rate=lr,
                        n_estimators=n_est,
                        random_state=42
                    )
                    m.fit(X.iloc[train_idx], y[train_idx])
                    pred = m.predict(X.iloc[val_idx])
                    mse = mean_squared_error(y[val_idx], pred)
                    rmse = mse ** 0.5
                    rmse_folds.append(rmse)
                mean_rmse = float(np.mean(rmse_folds))
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_model = GradientBoostingRegressor(
                        max_depth=depth,
                        learning_rate=lr,
                        n_estimators=n_est,
                        random_state=42
                    )
    best_model.fit(X, y)
    return {"model": best_model, "feature_names": list(X.columns), "best_rmse": float(best_rmse)}

def _recursive_forecast_gbr(model, last_row: pd.Series, horizon: int) -> np.ndarray:
    preds = []
    state = last_row.to_dict()
    for _ in range(horizon):
        X = np.array([[state[k] for k in model["feature_names"]]], dtype=float)
        yhat = model["model"].predict(X)[0]
        preds.append(yhat)

        lag_keys = [k for k in state.keys() if k.startswith("lag_")]
        lag_nums = sorted([int(k.split("_")[1]) for k in lag_keys], reverse=True)
        for n in lag_nums:
            if n == 1:
                state["lag_1"] = yhat
            else:
                state[f"lag_{n}"] = state.get(f"lag_{n-1}", state[f"lag_{n}"])
        if "roll12_mean" in state and "roll12_std" in state:
            state["roll12_mean"] = (state["roll12_mean"] * 12 - state.get("lag_12", yhat) + yhat) / 12.0
    return np.array(preds)

def ensemble_forecast(ts: pd.DataFrame, sarimax_res: Dict[str, Any], gbr_res: Dict[str, Any], horizon: int, use_log: bool=True) -> pd.DataFrame:
    idx = pd.date_range(ts.index[-1] + pd.offsets.MonthEnd(), periods=horizon, freq="M")

    sarimax_model = sarimax_res["model"]
    sarimax_fc = sarimax_model.get_forecast(steps=horizon)
    if use_log:
        mean_log = sarimax_fc.predicted_mean.values
        conf = sarimax_fc.conf_int(alpha=0.2).values
        mean = np.exp(mean_log)
        lower = np.exp(conf[:,0])
        upper = np.exp(conf[:,1])
    else:
        mean = sarimax_fc.predicted_mean.values
        conf = sarimax_fc.conf_int(alpha=0.2).values
        lower = conf[:,0]
        upper = conf[:,1]

    from .data_prep import make_lag_features
    feat_df = make_lag_features(ts["price_usd"])
    last_features = feat_df.iloc[-1]
    gbr_preds = _recursive_forecast_gbr(gbr_res, last_features, horizon)

    ens = 0.5 * np.array(mean) + 0.5 * gbr_preds
    out = pd.DataFrame({
        "forecast": ens,
        "sarimax": mean,
        "gbr": gbr_preds,
        "lower": lower,
        "upper": upper,
    }, index=idx)
    return out
