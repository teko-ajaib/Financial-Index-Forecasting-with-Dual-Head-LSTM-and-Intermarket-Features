import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

from ..config import ForecasterConfig
from ..results import TrainResult
from .backtest import sweep_confidence_thresholds, run_backtest


class ForwardTester:
    def __init__(self, config: ForecasterConfig):
        self.config = config

    def evaluate(self, train_result, data_dict, verbose=True):
        cfg = self.config
        model = train_result.model
        scaler = train_result.scaler
        device = train_result.device
        seq_len = train_result.hyperparameters.get("seq_len", cfg.seq_len)
        target_idx = train_result.target_idx

        model.eval()

        df_forward = data_dict["forward_test_df"]
        price_cols = [c for c in df_forward.columns if c in cfg._price_tickers]
        tech_cols = [c for c in df_forward.columns if c not in price_cols]

        df_rets = np.log(df_forward[price_cols]).diff().dropna()
        df_tech = df_forward[tech_cols].loc[df_rets.index]
        df_combined = pd.concat([df_rets, df_tech], axis=1)

        scaled_values = scaler.transform(df_combined.values)

        target_mean = scaler.mean_[target_idx]
        target_std = scaler.scale_[target_idx]
        target_col = cfg.target_ticker

        target_start = pd.to_datetime(cfg.forward_test_start)

        dates, actual_prices, prev_prices, pred_prices = [], [], [], []
        actual_rets, pred_rets, pred_probs = [], [], []

        with torch.no_grad():
            for i in range(seq_len, len(df_combined)):
                current_date = df_combined.index[i]
                if current_date < target_start:
                    continue

                inp = scaled_values[i - seq_len: i, :]
                tensor_in = torch.FloatTensor(inp).unsqueeze(0).to(device)
                p_reg, p_dir = model(tensor_in)

                norm_ret = p_reg.item()
                unscaled_ret = (norm_ret * target_std) + target_mean
                prob_up = torch.sigmoid(p_dir).item()

                actual_ret = df_combined.iloc[i, target_idx]
                cur_price = df_forward.loc[current_date, target_col]
                prev_date = df_combined.index[i - 1]
                prev_price = df_forward.loc[prev_date, target_col]
                predicted_price = prev_price * np.exp(unscaled_ret)

                dates.append(current_date)
                actual_prices.append(cur_price)
                prev_prices.append(prev_price)
                pred_prices.append(predicted_price)
                actual_rets.append(actual_ret)
                pred_rets.append(unscaled_ret)
                pred_probs.append(prob_up)

        df_results = pd.DataFrame({
            "Date": dates,
            "Actual_Price": actual_prices,
            "Predicted_Price": pred_prices,
            "Naive_Price": prev_prices,
            "Actual_Return": actual_rets,
            "Predicted_Return": pred_rets,
            "Pred_Prob_Up": pred_probs,
            "Price_Error": np.array(actual_prices) - np.array(pred_prices),
        }).set_index("Date")

        df_results["Actual_Direction"] = np.where(
            df_results["Actual_Price"] >= df_results["Naive_Price"], 1, 0
        )
        df_results["Pred_Direction_Raw"] = np.where(
            df_results["Pred_Prob_Up"] >= 0.50, 1, 0
        )
        df_results["Direction_Correct_Raw"] = np.where(
            df_results["Actual_Direction"] == df_results["Pred_Direction_Raw"], 1, 0
        )

        y_true = df_results["Actual_Price"].values
        y_pred = df_results["Predicted_Price"].values
        y_naive = df_results["Naive_Price"].values

        rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
        mae_model = mean_absolute_error(y_true, y_pred)
        mape_model = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2_model = r2_score(y_true, y_pred)

        rmse_naive = np.sqrt(mean_squared_error(y_true, y_naive))
        mae_naive = mean_absolute_error(y_true, y_naive)
        mape_naive = np.mean(np.abs((y_true - y_naive) / y_true)) * 100

        raw_acc = (df_results["Direction_Correct_Raw"].sum() / len(df_results)) * 100

        df_sweep, optimal_margin = sweep_confidence_thresholds(
            df_results, target_acc=cfg.target_directional_accuracy
        )
        chosen_margin = optimal_margin if optimal_margin is not None else cfg.confidence_margin

        probs = df_results["Pred_Prob_Up"].values
        trues = df_results["Actual_Direction"].values
        mask = (probs >= (0.50 + chosen_margin)) | (probs <= (0.50 - chosen_margin))

        high_acc = (
            ((probs[mask] >= 0.50).astype(int) == trues[mask]).mean() * 100
            if mask.sum() > 0 else raw_acc
        )
        cm_high = (
            confusion_matrix(trues[mask], (probs[mask] >= 0.50).astype(int), labels=[0, 1])
            if mask.sum() > 0 else None
        )

        bt = run_backtest(df_results, chosen_margin)

        metrics = {
            "n_test_days": len(df_results),
            "test_period": {
                "start": df_results.index.min().strftime("%Y-%m-%d"),
                "end": df_results.index.max().strftime("%Y-%m-%d"),
            },
            "price_regression": {
                "rmse_model": float(rmse_model),
                "mae_model": float(mae_model),
                "mape_model_pct": float(mape_model),
                "r2_model": float(r2_model),
                "rmse_naive": float(rmse_naive),
                "mae_naive": float(mae_naive),
                "mape_naive_pct": float(mape_naive),
            },
            "directional_accuracy": {
                "raw_all_days_pct": float(raw_acc),
                "optimal_confidence_margin": float(chosen_margin),
                "high_conviction_acc_pct": float(high_acc),
                "high_conviction_signals_count": int(mask.sum()),
                "high_conviction_coverage_pct": float((mask.sum() / len(df_results)) * 100),
            },
            "financial_backtest": bt,
            "confusion_matrix_high": cm_high.tolist() if cm_high is not None else None,
            "sweep_data": df_sweep,
        }

        if verbose:
            self._print_report(metrics)

        return metrics, df_results

    def _print_report(self, m):
        pr = m["price_regression"]
        da = m["directional_accuracy"]
        fb = m["financial_backtest"]

        print("\n" + "=" * 75)
        print("   FORWARD TEST REPORT (teko_ajaib)")
        print("=" * 75)
        print(f"Period: {m['test_period']['start']} to {m['test_period']['end']} "
              f"({m['n_test_days']} days)")
        print("-" * 75)
        print(f"{'Metric':<30}{'LSTM Model':<22}{'Baseline':<22}")
        print("-" * 75)
        print(f"{'RMSE':<30}${pr['rmse_model']:<21.2f}${pr['rmse_naive']:<21.2f}")
        print(f"{'MAE':<30}${pr['mae_model']:<21.2f}${pr['mae_naive']:<21.2f}")
        print(f"{'MAPE (%)':<30}{pr['mape_model_pct']:<21.3f}%{pr['mape_naive_pct']:<21.3f}%")
        print(f"{'R²':<30}{pr['r2_model']:<21.4f}")
        print(f"{'Dir. Acc (All Days)':<30}{da['raw_all_days_pct']:<21.2f}%50.00% (Random)")
        print(f"{'Dir. Acc (High-Conviction)':<30}{da['high_conviction_acc_pct']:<21.2f}%"
              f"Target >= {self.config.target_directional_accuracy}%")
        print("-" * 75)
        print(f"Active Signals: {da['high_conviction_signals_count']} "
              f"({da['high_conviction_coverage_pct']:.1f}% coverage)")
        print(f"Strategy Return: {fb['strategy_total_return_pct']:.2f}% "
              f"(vs Buy & Hold: {fb['buy_hold_total_return_pct']:.2f}%)")
        print(f"Sharpe: {fb['strategy_sharpe_ratio']:.2f} "
              f"(vs B&H: {fb['buy_hold_sharpe_ratio']:.2f})")
        print("=" * 75)
