import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ..config import ForecasterConfig
from .loader import download_data, load_from_csv, clean_and_impute
from .features import analyze_correlation, compute_technical_indicators


class DataEngineer:
    def __init__(self, config: ForecasterConfig):
        self.config = config

    def prepare(self, csv_path=None, download=True):
        cfg = self.config

        print("\n" + "=" * 75)
        print("        DATA ENGINEERING PIPELINE (teko_ajaib)")
        print("=" * 75)

        if csv_path is not None:
            df_raw = load_from_csv(csv_path)
        elif download:
            df_raw = download_data(
                cfg.candidate_tickers,
                start_date=cfg.start_train_date,
            )
        else:
            raise ValueError(
                "Provide csv_path or set download=True to download data."
            )

        df_clean = clean_and_impute(df_raw)

        df_train_val_period = df_clean.loc[cfg.start_train_date: cfg.split_date]
        print(f"\n[INFO] Modeling period: {cfg.start_train_date} to {cfg.split_date}")

        corr_matrix, target_corr, selected_features = analyze_correlation(
            df_train_val_period,
            target_col=cfg.target_ticker,
            threshold=cfg.correlation_threshold,
            include_vix=cfg.include_vix,
        )

        if cfg.use_technical_indicators:
            tech = compute_technical_indicators(df_clean, target_col=cfg.target_ticker)
            df_full = pd.concat(
                [df_clean[selected_features], tech], axis=1
            ).dropna()
            feature_names = list(df_full.columns)
        else:
            df_full = df_clean[selected_features].dropna()
            feature_names = list(selected_features)

        print(f"[INFO] Total final features: {len(feature_names)}")
        print(f"       {feature_names}")

        df_train_val = df_full.loc[cfg.start_train_date: cfg.split_date].copy()

        split_dt = pd.Timestamp(cfg.split_date)
        buffer_start = (split_dt - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        df_forward_test = df_full.loc[buffer_start:].copy()

        print(f"\n[INFO] Dataset Train-Val : {df_train_val.shape} "
              f"({df_train_val.index.min().strftime('%Y-%m-%d')} to "
              f"{df_train_val.index.max().strftime('%Y-%m-%d')})")
        print(f"[INFO] Dataset Forward   : {df_forward_test.shape} "
              f"({df_forward_test.index.min().strftime('%Y-%m-%d')} to "
              f"{df_forward_test.index.max().strftime('%Y-%m-%d')})")

        print("\n" + "=" * 75)
        print("        DATA ENGINEERING COMPLETE!")
        print("=" * 75)

        return {
            "raw_df": df_raw,
            "clean_df": df_clean,
            "train_val_df": df_train_val,
            "forward_test_df": df_forward_test,
            "full_features_df": df_full,
            "corr_matrix": corr_matrix,
            "target_corr": target_corr,
            "selected_features": selected_features,
            "feature_names": feature_names,
        }

    def save(self, data_dict, output_dir=None):
        out = self.config.get_output_dir() if output_dir is None else __import__("pathlib").Path(output_dir)
        (out / "data").mkdir(parents=True, exist_ok=True)

        data_dict["train_val_df"].to_csv(out / "data" / "train_val_data.csv")
        data_dict["forward_test_df"].to_csv(out / "data" / "forward_test_data.csv")
        data_dict["corr_matrix"].to_csv(out / "data" / "correlation_matrix.csv")

        print(f"[SAVE] Datasets saved at: {out / 'data'}")
