
import numpy as np
import pandas as pd

def load_and_prepare(df_or_path, date_col="date", price_col="price_usd", monthly=True, log_transform=True):
    if isinstance(df_or_path, (str, bytes)):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Expected columns: {date_col}, {price_col}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df[[date_col, price_col]].dropna()
    df = df[df[price_col] > 0]

    df = df.set_index(date_col)
    if monthly:
        df = df.resample("M").mean().ffill()

    if len(df) < 60:
        raise ValueError("Need at least ~60 monthly points for stable modeling.")

    if log_transform:
        df["price_usd_log"] = np.log(df[price_col])

    meta = {
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "points": int(len(df)),
        "freq": "M",
        "log_transform": bool(log_transform)
    }
    return df, meta


def make_lag_features(series, lags=[1,3,6,12,24,60]):
    """
    Build a supervised dataset from a price series with lagged features.
    """
    df = pd.DataFrame({"y": series})
    for L in lags:
        df[f"lag_{L}"] = series.shift(L)
    df["roll12_mean"] = series.rolling(12).mean()
    df["roll12_std"] = series.rolling(12).std()
    df = df.dropna()
    df = df.rename(columns={"y": "price_usd"})
    return df
