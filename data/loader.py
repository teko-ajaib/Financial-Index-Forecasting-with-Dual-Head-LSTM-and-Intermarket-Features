import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


def download_data(tickers, start_date="2015-01-01", end_date=None):
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "Package 'yfinance' is required for data download.\n"
            "Install with: pip install yfinance"
        )

    yf_tickers = []
    ticker_map = {}
    for t in tickers:
        if t.upper() == "VIX":
            yf_tickers.append("^VIX")
            ticker_map["^VIX"] = "VIX"
        else:
            yf_tickers.append(t)
            ticker_map[t] = t

    print(f"[INFO] Downloading data from Yahoo Finance: {yf_tickers}")
    print(f"       Range: {start_date} to {'present' if end_date is None else end_date}")

    data = yf.download(yf_tickers, start=start_date, end=end_date, auto_adjust=True)

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    elif "Close" in data.columns:
        data = data[["Close"]]
        data.columns = tickers[:1]

    rename_map = {}
    for col in data.columns:
        if col in ticker_map:
            rename_map[col] = ticker_map[col]
    if rename_map:
        data = data.rename(columns=rename_map)

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)

    available = [t for t in tickers if t in data.columns]
    missing = [t for t in tickers if t not in data.columns]
    if missing:
        print(f"[WARNING] The following tickers were not available: {missing}")

    data = data[available]
    print(f"[INFO] Successfully downloaded: {data.shape[0]} rows x {data.shape[1]} columns")
    return data


def load_from_csv(path, date_column="Date"):
    print(f"[INFO] Loading data from file: {path}")
    df = pd.read_csv(str(path), index_col=date_column, parse_dates=[date_column])
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)
    print(f"[INFO] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def clean_and_impute(df):
    n_before = df.isnull().sum().sum()
    df_clean = df.ffill().bfill()
    n_after = df_clean.isnull().sum().sum()
    if n_before > 0:
        print(f"[INFO] Missing values: {n_before} -> {n_after} (imputation complete)")
    return df_clean
