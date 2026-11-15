import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from ..config import ForecasterConfig
from ..results import TrainResult


class LiveForecaster:
    def __init__(self, config: ForecasterConfig):
        self.config = config

    def predict_next_week(self, train_result, data_dict, n_days=None, verbose=True):
        cfg = self.config
        if n_days is None:
            n_days = getattr(cfg, "forecast_days", 5)
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

        var_qqq = df_rets[cfg.target_ticker].var()
        betas = {
            c: float(df_rets[c].cov(df_rets[cfg.target_ticker]) / (var_qqq + 1e-9))
            for c in price_cols
        }
        daily_vol = float(df_rets[cfg.target_ticker].std())

        target_mean = scaler.mean_[target_idx]
        target_std = scaler.scale_[target_idx]

        last_date = df_combined.index[-1]
        last_price = float(df_forward.loc[last_date, cfg.target_ticker])
        future_dates = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=n_days
        )

        curr_window = df_combined.iloc[-seq_len:].copy()
        curr_price = last_price
        records = []

        if verbose:
            print("\n" + "=" * 80)
            print(f"  LIVE FORECAST: {cfg.target_ticker} "
                  f"({future_dates[0].strftime('%d %b %Y')} to "
                  f"{future_dates[-1].strftime('%d %b %Y')})")
            print("=" * 80)
            print(f"Last Observed Price: ${last_price:.2f} "
                  f"({last_date.strftime('%Y-%m-%d')})")
            print("-" * 80)

        for step, f_date in enumerate(future_dates, start=1):
            scaled_input = scaler.transform(curr_window.values)
            tensor_in = torch.FloatTensor(scaled_input).unsqueeze(0).to(device)

            with torch.no_grad():
                p_reg, p_dir = model(tensor_in)

            unscaled_ret = (p_reg.item() * target_std) + target_mean
            prob_up = torch.sigmoid(p_dir).item()

            pred_price = curr_price * np.exp(unscaled_ret)
            conf_lvl = getattr(cfg, "confidence_level", 0.95)
            z_score = 1.96 if abs(conf_lvl - 0.95) < 1e-3 else (2.576 if abs(conf_lvl - 0.99) < 1e-3 else (1.645 if abs(conf_lvl - 0.90) < 1e-3 else 1.96))
            ci_band = z_score * daily_vol * np.sqrt(step)
            lower = pred_price * np.exp(-ci_band)
            upper = pred_price * np.exp(ci_band)

            is_bull = prob_up >= 0.50
            dir_label = "BULLISH" if is_bull else "BEARISH"
            margin = abs(prob_up - 0.50)
            if margin >= 0.15:
                signal = "Strong Signal"
            elif margin >= 0.05:
                signal = "Moderate Signal"
            else:
                signal = "Consolidation"

            day_name = f_date.strftime("%A")

            records.append({
                "Step": step,
                "Date": f_date.strftime("%Y-%m-%d"),
                "Day": day_name,
                "Predicted_Price": round(pred_price, 2),
                "Lower_Bound_95": round(lower, 2),
                "Upper_Bound_95": round(upper, 2),
                "Daily_Return_Pct": round(unscaled_ret * 100, 2),
                "Direction": dir_label,
                "Bullish_Prob_Pct": round(prob_up * 100, 1),
                "Bearish_Prob_Pct": round((1.0 - prob_up) * 100, 1),
                "Signal_Strength": signal,
            })

            new_row = {}
            for c in price_cols:
                new_row[c] = unscaled_ret * betas[c]

            if "RSI" in curr_window.columns:
                new_row["RSI"] = np.clip(
                    curr_window["RSI"].iloc[-1] + (unscaled_ret * 1.5), 0.1, 0.9
                )
            if "MACD_HIST" in curr_window.columns:
                new_row["MACD_HIST"] = (
                    curr_window["MACD_HIST"].iloc[-1] * 0.85 + unscaled_ret * 0.15
                )
            if "MOM3" in curr_window.columns:
                new_row["MOM3"] = unscaled_ret
            if "MOM5" in curr_window.columns:
                new_row["MOM5"] = unscaled_ret
            if "SMA5_RATIO" in curr_window.columns:
                new_row["SMA5_RATIO"] = (
                    curr_window["SMA5_RATIO"].iloc[-1] * 0.8 + unscaled_ret * 0.2
                )
            if "SMA20_RATIO" in curr_window.columns:
                new_row["SMA20_RATIO"] = (
                    curr_window["SMA20_RATIO"].iloc[-1] * 0.9 + unscaled_ret * 0.1
                )

            new_series = pd.DataFrame([new_row], index=[f_date])
            curr_window = pd.concat([curr_window.iloc[1:], new_series])
            curr_price = pred_price

        df_forecast = pd.DataFrame(records)

        if verbose:
            print(df_forecast.to_string(index=False))
            print("-" * 80)
            total_chg = ((records[-1]["Predicted_Price"] - last_price) / last_price) * 100
            print(f"Start: ${last_price:.2f} -> Target: ${records[-1]['Predicted_Price']:.2f} "
                  f"({total_chg:+.2f}%)")
            print("=" * 80)

        summary = {
            "last_observed_date": last_date.strftime("%Y-%m-%d"),
            "last_observed_price": last_price,
            "forecast_start": future_dates[0].strftime("%Y-%m-%d"),
            "forecast_end": future_dates[-1].strftime("%Y-%m-%d"),
            "end_target_price": records[-1]["Predicted_Price"],
            "weekly_change_pct": round(
                ((records[-1]["Predicted_Price"] - last_price) / last_price) * 100, 2
            ),
            "support_95": float(df_forecast["Lower_Bound_95"].min()),
            "resistance_95": float(df_forecast["Upper_Bound_95"].max()),
            "daily_forecasts": records,
        }

        return df_forecast, summary
