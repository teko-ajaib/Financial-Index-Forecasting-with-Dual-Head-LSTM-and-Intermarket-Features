import time
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from ..config import ForecasterConfig
from ..model import DualHeadLSTM, TimeSeriesSequenceDataset
from ..results import TrainResult
from .loss import DirectionAwareJointLoss


def _set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(self, config: ForecasterConfig, seed=None):
        self.config = config
        self.seed = seed if seed is not None else getattr(config, "seed", 42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _set_seed(self.seed)

    def _prepare_datasets(self, data_dict):
        cfg = self.config
        df_features = data_dict["train_val_df"]

        price_cols = [c for c in df_features.columns if c in cfg._price_tickers]
        tech_cols = [c for c in df_features.columns if c not in price_cols]

        df_rets = np.log(df_features[price_cols]).diff().dropna()
        df_tech = df_features[tech_cols].loc[df_rets.index]
        df_combined = pd.concat([df_rets, df_tech], axis=1)

        n_total = len(df_combined)
        n_val = int(np.ceil(n_total * cfg.val_split_ratio))
        train_end = n_total - n_val

        df_train = df_combined.iloc[:train_end]
        df_val = df_combined.iloc[train_end:]

        print(f"[INFO] Train Set : {len(df_train)} days "
              f"({df_train.index.min().strftime('%Y-%m-%d')} to "
              f"{df_train.index.max().strftime('%Y-%m-%d')})")
        print(f"[INFO] Val Set   : {len(df_val)} days "
              f"({df_val.index.min().strftime('%Y-%m-%d')} to "
              f"{df_val.index.max().strftime('%Y-%m-%d')})")

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(df_train.values)

        val_with_buffer = df_combined.iloc[train_end - cfg.seq_len:].values
        val_scaled = scaler.transform(val_with_buffer)

        raw_rets_train = df_train.values
        raw_rets_val = df_combined.iloc[train_end - cfg.seq_len:].values

        train_dataset = TimeSeriesSequenceDataset(
            train_scaled, raw_returns=raw_rets_train,
            seq_len=cfg.seq_len, target_idx=0,
        )
        val_dataset = TimeSeriesSequenceDataset(
            val_scaled, raw_returns=raw_rets_val,
            seq_len=cfg.seq_len, target_idx=0,
        )

        print(f"[INFO] Sequences: Train={len(train_dataset)}, Val={len(val_dataset)} "
              f"(lookback={cfg.seq_len})")

        return {
            "df_combined": df_combined,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "scaler": scaler,
            "feature_names": list(df_combined.columns),
            "target_col": cfg.target_ticker,
        }

    def train(self, data_dict, params=None, verbose=True):
        _set_seed(self.seed)
        cfg = self.config
        hp = params if params else cfg.get_hyperparams()

        print(f"\n[INFO] Device: {self.device}")

        prepared = self._prepare_datasets(data_dict)
        train_dataset = prepared["train_dataset"]
        val_dataset = prepared["val_dataset"]
        feature_names = prepared["feature_names"]
        scaler = prepared["scaler"]
        num_features = len(feature_names)

        train_loader = DataLoader(
            train_dataset, batch_size=hp.get("batch_size", 32), shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=hp.get("batch_size", 32), shuffle=False
        )

        model = DualHeadLSTM(
            input_size=num_features,
            hidden_size=hp.get("hidden_size", 64),
            num_layers=hp.get("num_layers", 2),
            dropout=hp.get("dropout", 0.2),
        ).to(self.device)

        criterion = DirectionAwareJointLoss(
            alpha=hp.get("loss_alpha", getattr(cfg, "loss_alpha", 1.0)),
            beta=hp.get("loss_beta", getattr(cfg, "loss_beta", 0.5)),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=hp.get("learning_rate", 0.001),
            weight_decay=hp.get("weight_decay", 1e-3),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=4
        )

        num_epochs = hp.get("num_epochs", 40)
        patience = hp.get("early_stopping_patience", 12)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_model_state = None
        best_epoch = 0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        if verbose:
            print(f"\n[TRAIN] Dual-Head Direction-Aware LSTM")
            print(f"        Hidden={hp['hidden_size']} | Layers={hp['num_layers']} | "
                  f"LR={hp['learning_rate']} | SeqLen={hp.get('seq_len', cfg.seq_len)}")
            print("-" * 70)
            print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Loss':<14}{'Val Acc (%)':<14}{'Status':<14}")
            print("-" * 70)

        start = time.time()

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss = 0.0
            for X_b, y_reg_b, y_dir_b in train_loader:
                X_b = X_b.to(self.device)
                y_reg_b = y_reg_b.to(self.device)
                y_dir_b = y_dir_b.to(self.device)

                optimizer.zero_grad()
                p_reg, p_dir = model(X_b)
                loss, _, _ = criterion(p_reg, p_dir, y_reg_b, y_dir_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_b)

            train_loss /= len(train_dataset)

            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for X_b, y_reg_b, y_dir_b in val_loader:
                    X_b = X_b.to(self.device)
                    y_reg_b = y_reg_b.to(self.device)
                    y_dir_b = y_dir_b.to(self.device)

                    p_reg, p_dir = model(X_b)
                    loss, _, _ = criterion(p_reg, p_dir, y_reg_b, y_dir_b)
                    val_loss += loss.item() * len(X_b)

                    pred_bin = (torch.sigmoid(p_dir) >= 0.50).float()
                    correct += (pred_bin == y_dir_b).sum().item()

            val_loss /= len(val_dataset)
            val_acc = (correct / len(val_dataset)) * 100
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                status = "* Best"
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    status = f"Early Stop ({patience_counter}/{patience})"
                    if verbose:
                        print(f"{epoch:<8}{train_loss:<14.4f}{val_loss:<14.4f}"
                              f"{val_acc:<14.2f}{status:<14}")
                    break

            if verbose and (epoch % 5 == 0 or epoch == 1 or status):
                print(f"{epoch:<8}{train_loss:<14.4f}{val_loss:<14.4f}"
                      f"{val_acc:<14.2f}{status:<14}")

        elapsed = time.time() - start
        if verbose:
            print("-" * 70)
            print(f"[DONE] {elapsed:.1f}s | Best Epoch={best_epoch} | "
                  f"Val Loss={best_val_loss:.4f} | Val Acc={best_val_acc:.2f}%")

        model.load_state_dict(best_model_state)
        model.to(self.device)
        model.eval()

        return TrainResult(
            model=model,
            scaler=scaler,
            history=history,
            feature_names=feature_names,
            num_features=num_features,
            target_ticker=cfg.target_ticker,
            target_idx=0,
            hyperparameters=hp,
            best_val_loss=best_val_loss,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            device=self.device,
        )

    def optimize(self, data_dict, param_grid=None, verbose=True):
        if param_grid is None:
            param_grid = [
                {"hidden_size": 64, "num_layers": 2, "learning_rate": 0.001,
                 "dropout": 0.2, "batch_size": 32, "num_epochs": 20, "seq_len": 10,
                 "early_stopping_patience": 12, "weight_decay": 1e-3, "val_split_ratio": 0.20},
                {"hidden_size": 64, "num_layers": 1, "learning_rate": 0.001,
                 "dropout": 0.1, "batch_size": 32, "num_epochs": 20, "seq_len": 10,
                 "early_stopping_patience": 12, "weight_decay": 1e-3, "val_split_ratio": 0.20},
                {"hidden_size": 128, "num_layers": 2, "learning_rate": 0.0005,
                 "dropout": 0.2, "batch_size": 32, "num_epochs": 25, "seq_len": 10,
                 "early_stopping_patience": 12, "weight_decay": 1e-3, "val_split_ratio": 0.20},
                {"hidden_size": 32, "num_layers": 2, "learning_rate": 0.001,
                 "dropout": 0.1, "batch_size": 32, "num_epochs": 20, "seq_len": 10,
                 "early_stopping_patience": 12, "weight_decay": 1e-3, "val_split_ratio": 0.20},
            ]

        print("\n" + "=" * 70)
        print("   HYPERPARAMETER OPTIMIZATION (DIRECTIONAL ACCURACY)")
        print("=" * 70)

        results = []
        for idx, params in enumerate(param_grid, start=1):
            print(f"\n>>> Trial [{idx}/{len(param_grid)}]: {params}")
            res = self.train(data_dict, params=params, verbose=False)
            print(f"    Val Acc={res.best_val_acc:.2f}% | Val Loss={res.best_val_loss:.4f}")
            results.append({
                "trial": idx, "params": params,
                "best_val_acc": res.best_val_acc,
                "best_val_loss": res.best_val_loss,
            })

        df_results = pd.DataFrame([{
            "Trial": r["trial"],
            "Hidden": r["params"]["hidden_size"],
            "Layers": r["params"]["num_layers"],
            "LR": r["params"]["learning_rate"],
            "Val Acc (%)": round(r["best_val_acc"], 2),
            "Val Loss": round(r["best_val_loss"], 4),
        } for r in results]).sort_values("Val Acc (%)", ascending=False)

        print("\n=== OPTIMIZATION RANKING ===")
        print(df_results.to_string(index=False))

        best_idx = int(df_results.iloc[0]["Trial"])
        best_params = [r["params"] for r in results if r["trial"] == best_idx][0]
        return best_params, df_results

    def save(self, result, path=None):
        from pathlib import Path as P
        out = P(path) if path else self.config.get_output_dir() / "models"
        out.mkdir(parents=True, exist_ok=True)

        torch.save(result.model.state_dict(), out / "best_lstm_model.pth")
        joblib.dump(result.scaler, out / "scaler.pkl")

        meta = {
            "model_architecture": "DualHeadLSTM",
            "input_features": result.feature_names,
            "num_features": result.num_features,
            "target_ticker": result.target_ticker,
            "target_idx": result.target_idx,
            "hyperparameters": {
                k: (float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in result.hyperparameters.items()
            },
            "best_epoch": result.best_epoch,
            "best_val_loss": float(result.best_val_loss),
            "best_val_acc": float(result.best_val_acc),
        }
        with open(out / "training_config.json", "w") as f:
            json.dump(meta, f, indent=4)

        print(f"[SAVE] Model, scaler, config saved at: {out}")

    @staticmethod
    def load(path, config=None):
        from pathlib import Path as P
        p = P(path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(p / "training_config.json", "r") as f:
            meta = json.load(f)

        scaler = joblib.load(p / "scaler.pkl")

        hp = meta["hyperparameters"] if "hyperparameters" in meta else meta
        model = DualHeadLSTM(
            input_size=meta["num_features"],
            hidden_size=hp["hidden_size"],
            num_layers=hp["num_layers"],
            dropout=hp["dropout"],
        )
        model.load_state_dict(torch.load(p / "best_lstm_model.pth", map_location=device))
        model.to(device)
        model.eval()

        print(f"[LOAD] Model loaded from: {p}")

        return TrainResult(
            model=model,
            scaler=scaler,
            history={},
            feature_names=meta.get("input_features", []),
            num_features=meta["num_features"],
            target_ticker=meta.get("target_ticker", "QQQ"),
            target_idx=meta.get("target_idx", 0),
            hyperparameters=hp,
            best_val_loss=meta.get("best_val_loss", 0),
            best_val_acc=meta.get("best_val_acc", 0),
            best_epoch=meta.get("best_epoch", 0),
            device=device,
        )
