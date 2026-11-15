import numpy as np
import pandas as pd


def sweep_confidence_thresholds(df_results, target_acc=70.0):
    probs = df_results["Pred_Prob_Up"].values
    trues = df_results["Actual_Direction"].values

    sweep_data = []
    optimal_margin = None

    for margin in np.arange(0.00, 0.25, 0.01):
        mask = (probs >= (0.50 + margin)) | (probs <= (0.50 - margin))
        n_signals = mask.sum()
        if n_signals >= 5:
            pred_dir = (probs[mask] >= 0.50).astype(int)
            acc = (pred_dir == trues[mask]).mean() * 100
            cov = (n_signals / len(probs)) * 100

            sweep_data.append({
                "margin": float(margin),
                "lower_bound": round(0.50 - margin, 4),
                "upper_bound": round(0.50 + margin, 4),
                "n_signals": int(n_signals),
                "coverage_pct": round(cov, 2),
                "directional_acc_pct": round(acc, 2),
            })

            if acc >= target_acc and optimal_margin is None and n_signals >= 10:
                optimal_margin = margin

    df_sweep = pd.DataFrame(sweep_data)
    return df_sweep, optimal_margin


def run_backtest(df_results, chosen_margin=0.05):
    probs = df_results["Pred_Prob_Up"].values
    actual_returns = df_results["Actual_Return"].values

    positions = np.zeros(len(df_results))
    positions[probs >= (0.50 + chosen_margin)] = 1.0
    positions[probs <= (0.50 - chosen_margin)] = -1.0

    strat_returns = positions * actual_returns
    cum_strategy = np.exp(np.cumsum(strat_returns)) - 1.0
    cum_buy_hold = np.exp(np.cumsum(actual_returns)) - 1.0

    sharpe_strat = (np.mean(strat_returns) / (np.std(strat_returns) + 1e-9)) * np.sqrt(252)
    sharpe_bh = (np.mean(actual_returns) / (np.std(actual_returns) + 1e-9)) * np.sqrt(252)

    return {
        "strategy_total_return_pct": float(cum_strategy[-1] * 100),
        "buy_hold_total_return_pct": float(cum_buy_hold[-1] * 100),
        "strategy_sharpe_ratio": float(sharpe_strat),
        "buy_hold_sharpe_ratio": float(sharpe_bh),
        "cum_strategy": cum_strategy,
        "cum_buy_hold": cum_buy_hold,
    }
