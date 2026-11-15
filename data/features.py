import numpy as np
import pandas as pd


def analyze_correlation(df, target_col, threshold=0.85, include_vix=True):
    corr_matrix = df.corr()
    target_corr = corr_matrix[target_col].sort_values(ascending=False)

    selected = []
    ranking = []

    for ticker, r in target_corr.items():
        is_target = (ticker == target_col)
        meets = abs(r) >= threshold

        status = "Primary Target" if is_target else ("SELECTED" if meets else "SKIPPED")
        if meets or is_target:
            selected.append(ticker)

        ranking.append({"Ticker": ticker, "Correlation": round(r, 4), "Status": status})

    if include_vix and "VIX" not in selected and "VIX" in df.columns:
        selected.append("VIX")
        print("[INFO] VIX included as market sentiment risk indicator.")

    if target_col in selected:
        selected.remove(target_col)
        selected = [target_col] + selected

    print(f"\n=== CORRELATION RANKING TO {target_col} ===")
    df_rank = pd.DataFrame(ranking)
    print(df_rank.to_string(index=False))
    print(f"\n[INFO] Selected features ({len(selected)}): {selected}")

    return corr_matrix, target_corr, selected


def compute_technical_indicators(df_price, target_col="QQQ"):
    close = df_price[target_col]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss_val = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss_val.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = (100 - (100 / (1 + rs))) / 100.0

    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = (exp12 - exp26) / close
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal

    mom3 = close.pct_change(3)
    mom5 = close.pct_change(5)
    sma5_ratio = close / close.rolling(5).mean() - 1.0
    sma20_ratio = close / close.rolling(20).mean() - 1.0

    indicators = pd.DataFrame({
        "RSI": rsi,
        "MACD_HIST": macd_hist,
        "MOM3": mom3,
        "MOM5": mom5,
        "SMA5_RATIO": sma5_ratio,
        "SMA20_RATIO": sma20_ratio,
    }, index=df_price.index)

    print("[INFO] 6 technical indicators calculated: RSI, MACD_HIST, MOM3, MOM5, SMA5_RATIO, SMA20_RATIO")
    return indicators
