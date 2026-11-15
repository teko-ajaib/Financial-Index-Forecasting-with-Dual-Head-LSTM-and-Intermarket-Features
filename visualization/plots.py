import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def _save_or_show(fig, save_path):
    plt.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved: {save_path}")
    else:
        plt.show()


def plot_correlation_heatmap(corr_matrix, selected_features=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    if selected_features:
        ordered = [c for c in selected_features if c in corr_matrix.columns]
        remaining = [c for c in corr_matrix.columns if c not in ordered]
        order = ordered + remaining
        sub = corr_matrix.loc[order, order]
    else:
        sub = corr_matrix
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"label": "Pearson Correlation (r)"}, ax=ax)
    ax.set_title("Intermarket Asset Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)
    _save_or_show(fig, save_path)


def plot_dataset_overview(df_clean, selected_features, split_date,
                          target_ticker="QQQ", start_date="2015-01-01", save_path=None):
    price_cols = [c for c in selected_features if c in df_clean.columns and c != "VIX"]
    df_slice = df_clean.loc[start_date:][price_cols]
    df_norm = (df_slice / df_slice.iloc[0]) * 100

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    ax.plot(df_norm.index, df_norm[target_ticker],
            label=f"{target_ticker} (Target)", color="#0B5394", linewidth=2.5)
    for col in price_cols:
        if col != target_ticker:
            ax.plot(df_norm.index, df_norm[col], label=col, alpha=0.5, linewidth=1.1)

    split_dt = pd.to_datetime(split_date)
    ax.axvline(split_dt, color="#C0392B", linestyle="--", linewidth=2.0,
               label=f"Split Boundary ({split_date})")

    ax.set_title("Normalized Price Dynamics & Dataset Partition",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontweight="bold")
    ax.set_ylabel("Normalized Price (Base 100)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    _save_or_show(fig, save_path)


def plot_training_curve(history, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#2980B9", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#E74C3C", lw=2, ls="--")
    ax1.set_title("Joint Loss (Train vs Val)", fontweight="bold")
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Loss", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="Val Directional Acc (%)", color="#27AE60", lw=2)
    ax2.axhline(50.0, color="black", linestyle=":", label="Random Baseline (50%)")
    ax2.set_title("Validation Directional Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    _save_or_show(fig, save_path)


def plot_forward_test(df_results, metrics=None, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=150, sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})

    ax1.plot(df_results.index, df_results["Actual_Price"],
             label="Actual Price ($)", color="#1F618D", lw=2.2)
    ax1.plot(df_results.index, df_results["Predicted_Price"],
             label="Predicted LSTM ($)", color="#E67E22", lw=2.0, ls="--")

    title = "Forward Test: Actual vs Predicted"
    if metrics:
        pr = metrics.get("price_regression", {})
        da = metrics.get("directional_accuracy", {})
        title += (f"\n[RMSE: ${pr.get('rmse_model', 0):.2f} | "
                  f"Dir Acc: {da.get('high_conviction_acc_pct', 0):.1f}%]")
    ax1.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Price ($)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(df_results.index, df_results["Price_Error"], color="#C0392B", lw=1.2)
    ax2.axhline(0, color="black", ls="--", lw=1.0)
    ax2.fill_between(df_results.index, df_results["Price_Error"], 0,
                     color="#C0392B", alpha=0.15)
    ax2.set_ylabel("Residual Error ($)", fontweight="bold")
    ax2.set_xlabel("Date", fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    _save_or_show(fig, save_path)


def plot_confidence_curve(df_sweep, target_acc=70.0, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.plot(df_sweep["margin"], df_sweep["directional_acc_pct"],
            marker="o", color="#27AE60", lw=2.2, label="Directional Accuracy (%)")
    ax.axhline(target_acc, color="#C0392B", ls="--", lw=1.8,
               label=f"Target ({target_acc}%)")
    ax.axhline(50.0, color="gray", ls=":", label="Random (50%)")
    ax.set_title("Confidence Margin Optimization vs Directional Accuracy",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Confidence Margin (|P - 0.50|)", fontweight="bold")
    ax.set_ylabel("Directional Accuracy (%)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    _save_or_show(fig, save_path)


def plot_confusion_matrix(cm, acc=None, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cm_df = pd.DataFrame(cm, index=["Actual Down", "Actual Up"],
                         columns=["Pred Down", "Pred Up"])
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Greens", cbar=False,
                annot_kws={"size": 14}, ax=ax)
    title = "Market Direction Confusion Matrix (High-Conviction)"
    if acc:
        title += f"\nHit Rate: {acc:.2f}%"
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_ylabel("Actual Label", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")
    _save_or_show(fig, save_path)


def plot_backtest(df_results, cum_strategy, cum_buy_hold,
                  strategy_return=None, bh_return=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    s_label = f"LSTM Strategy ({strategy_return:.1f}%)" if strategy_return else "LSTM Strategy"
    b_label = f"Buy & Hold ({bh_return:.1f}%)" if bh_return else "Buy & Hold"

    ax.plot(df_results.index, cum_strategy * 100, label=s_label,
            color="#27AE60", lw=2.2)
    ax.plot(df_results.index, cum_buy_hold * 100, label=b_label,
            color="#7F8C8D", lw=1.8, ls="--")
    ax.axhline(0, color="black", ls=":", alpha=0.7)
    ax.set_title("Cumulative Backtest: Model Strategy vs Buy & Hold", fontweight="bold", pad=12)
    ax.set_ylabel("Cumulative Return (%)", fontweight="bold")
    ax.set_xlabel("Date", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    _save_or_show(fig, save_path)


def plot_live_forecast(df_forward_prices, df_forecast, last_date, last_price,
                       target_ticker="QQQ", save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), dpi=150,
                                    gridspec_kw={"height_ratios": [2.2, 1]})

    hist_tail = df_forward_prices.loc[
        last_date - pd.Timedelta(days=50): last_date
    ]
    if target_ticker in hist_tail.columns:
        ax1.plot(hist_tail.index, hist_tail[target_ticker],
                 label="Actual Historical Price ($)", color="#1B4F72", lw=2.4)

    fc_dates = [last_date] + list(pd.to_datetime(df_forecast["Date"]))
    fc_prices = [last_price] + list(df_forecast["Predicted_Price"])
    fc_lower = [last_price] + list(df_forecast["Lower_Bound_95"])
    fc_upper = [last_price] + list(df_forecast["Upper_Bound_95"])

    ax1.plot(fc_dates, fc_prices, label="LSTM Forecast", color="#E67E22",
             lw=2.4, ls="--", marker="o", ms=6)
    ax1.fill_between(fc_dates, fc_lower, fc_upper, color="#F39C12", alpha=0.18,
                     label="95% Confidence Band")
    ax1.set_title(f"Live Multi-Step Forecast: {target_ticker} (Next Week)",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Price ($)", fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    bar_dates = pd.to_datetime(df_forecast["Date"])
    rets = df_forecast["Daily_Return_Pct"]
    colors = ["#27AE60" if r >= 0 else "#E74C3C" for r in rets]

    bars = ax2.bar(bar_dates, rets, color=colors, width=0.6, alpha=0.85,
                   edgecolor="black", lw=0.8)
    ax2.axhline(0, color="black", ls="--", lw=1.0)

    for bar, r in zip(bars, rets):
        y = bar.get_height() + (0.1 if r >= 0 else -0.3)
        ax2.text(bar.get_x() + bar.get_width() / 2, y, f"{r:+.2f}%",
                 ha="center", va="bottom" if r >= 0 else "top",
                 fontsize=9, fontweight="bold")

    ax2.set_ylabel("Daily Return (%)", fontweight="bold")
    ax2.set_xlabel("Trading Day", fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax2.grid(True, linestyle=":", alpha=0.6)

    _save_or_show(fig, save_path)
