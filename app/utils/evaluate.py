
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def backtest_multi_step(series: pd.Series, build_fn: Callable[[pd.Series], Dict[str, Any]], horizon: int=12, n_splits: int=3) -> pd.DataFrame:
    """
    Rolling-origin evaluation: for each split, fit on train, predict next 'horizon' steps, compare to truth.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    y = series.values
    for i, (train_idx, val_idx) in enumerate(tscv.split(y)):
        train = series.iloc[train_idx]
        test = series.iloc[val_idx]

        model_res = build_fn(train)
        steps = min(horizon, len(test))
        fc = model_res["model"].get_forecast(steps=steps).predicted_mean.values
        real = test.values[:steps]

        if series.name and "log" in series.name:
            fc = np.exp(fc)
            real = np.exp(real)

        mape = mean_absolute_percentage_error(real, fc)
        mse = mean_squared_error(real, fc)
        rmse = mse ** 0.5

        rows.append({"fold": i+1, "len_train": len(train), "len_test": len(test), "h_eval": int(steps), "MAPE": float(mape), "RMSE": float(rmse)})
    return pd.DataFrame(rows)
