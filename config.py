import json
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from pathlib import Path


@dataclass
class ForecasterConfig:
    target_ticker: str = "QQQ"
    candidate_tickers: List[str] = field(default_factory=lambda: [
        "QQQ", "AAPL", "NVDA", "MSFT", "AMZN",
        "AMD", "MU", "TSLA", "META", "GOOGL", "AVGO", "VIX",
    ])

    start_train_date: str = "2015-01-01"
    split_date: str = "2026-06-30"
    forward_test_start: str = "2026-07-01"

    correlation_threshold: float = 0.85
    use_technical_indicators: bool = True
    include_vix: bool = True
    target_directional_accuracy: float = 70.0
    confidence_margin: float = 0.05

    seq_len: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    weight_decay: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 40
    early_stopping_patience: int = 12
    val_split_ratio: float = 0.20

    loss_alpha: float = 1.0
    loss_beta: float = 0.5
    seed: int = 42

    forecast_days: int = 5
    confidence_level: float = 0.95

    output_dir: Optional[str] = None
    data_dir: Optional[str] = None
    raw_data_path: Optional[str] = None

    _price_tickers: List[str] = field(default_factory=lambda: [
        "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN",
        "META", "TSLA", "AMD", "NVDA", "AVGO", "VIX",
    ])

    def get_output_dir(self) -> Path:
        p = Path(self.output_dir) if self.output_dir else Path("./forecast_output")
        for sub in ["data", "models", "reports"]:
            (p / sub).mkdir(parents=True, exist_ok=True)
        return p

    def get_hyperparams(self) -> dict:
        return {
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "val_split_ratio": self.val_split_ratio,
            "seed": self.seed,
            "loss_alpha": self.loss_alpha,
            "loss_beta": self.loss_beta,
        }

    def to_dict(self, nested: bool = True) -> Dict[str, Any]:
        if not nested:
            return {
                "target_ticker": self.target_ticker,
                "candidate_tickers": list(self.candidate_tickers),
                "start_train_date": self.start_train_date,
                "split_date": self.split_date,
                "forward_test_start": self.forward_test_start,
                "correlation_threshold": self.correlation_threshold,
                "use_technical_indicators": self.use_technical_indicators,
                "include_vix": self.include_vix,
                "target_directional_accuracy": self.target_directional_accuracy,
                "confidence_margin": self.confidence_margin,
                "seq_len": self.seq_len,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "val_split_ratio": self.val_split_ratio,
                "loss_alpha": self.loss_alpha,
                "loss_beta": self.loss_beta,
                "seed": self.seed,
                "forecast_days": self.forecast_days,
                "confidence_level": self.confidence_level,
                "output_dir": self.output_dir,
                "data_dir": self.data_dir,
                "raw_data_path": self.raw_data_path,
                "price_tickers": list(self._price_tickers),
            }

        return {
            "project": {
                "name": "teko-ajaib",
                "version": "1.0.0",
            },
            "data": {
                "target_ticker": self.target_ticker,
                "candidate_tickers": list(self.candidate_tickers),
                "price_tickers": list(self._price_tickers),
                "start_train_date": self.start_train_date,
                "split_date": self.split_date,
                "forward_test_start": self.forward_test_start,
                "correlation_threshold": self.correlation_threshold,
                "use_technical_indicators": self.use_technical_indicators,
                "include_vix": self.include_vix,
            },
            "model": {
                "architecture": "DualHeadLSTM",
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
            "training": {
                "seq_len": self.seq_len,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "early_stopping_patience": self.early_stopping_patience,
                "val_split_ratio": self.val_split_ratio,
                "seed": self.seed,
            },
            "loss": {
                "alpha": self.loss_alpha,
                "beta": self.loss_beta,
            },
            "evaluation": {
                "target_directional_accuracy": self.target_directional_accuracy,
                "confidence_margin": self.confidence_margin,
            },
            "forecasting": {
                "forecast_days": self.forecast_days,
                "confidence_level": self.confidence_level,
            },
            "paths": {
                "output_dir": self.output_dir,
                "data_dir": self.data_dir,
                "raw_data_path": self.raw_data_path,
            },
        }

    def to_json(self, json_path: Union[str, Path] = "config.json", indent: int = 4) -> Path:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(nested=True), f, indent=indent)
        print(f"[CONFIG] Configuration saved to: {p}")
        return p

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForecasterConfig":
        kwargs: Dict[str, Any] = {}

        if "data" in data and isinstance(data["data"], dict):
            d = data["data"]
            if "target_ticker" in d: kwargs["target_ticker"] = str(d["target_ticker"])
            if "candidate_tickers" in d: kwargs["candidate_tickers"] = list(d["candidate_tickers"])
            if "price_tickers" in d: kwargs["_price_tickers"] = list(d["price_tickers"])
            if "start_train_date" in d: kwargs["start_train_date"] = str(d["start_train_date"])
            if "split_date" in d: kwargs["split_date"] = str(d["split_date"])
            if "forward_test_start" in d: kwargs["forward_test_start"] = str(d["forward_test_start"])
            if "correlation_threshold" in d: kwargs["correlation_threshold"] = float(d["correlation_threshold"])
            if "use_technical_indicators" in d: kwargs["use_technical_indicators"] = bool(d["use_technical_indicators"])
            if "include_vix" in d: kwargs["include_vix"] = bool(d["include_vix"])

        if "model" in data and isinstance(data["model"], dict):
            m = data["model"]
            if "hidden_size" in m: kwargs["hidden_size"] = int(m["hidden_size"])
            if "num_layers" in m: kwargs["num_layers"] = int(m["num_layers"])
            if "dropout" in m: kwargs["dropout"] = float(m["dropout"])
            if "seq_len" in m: kwargs["seq_len"] = int(m["seq_len"])

        if "training" in data and isinstance(data["training"], dict):
            t = data["training"]
            if "seq_len" in t: kwargs["seq_len"] = int(t["seq_len"])
            if "batch_size" in t: kwargs["batch_size"] = int(t["batch_size"])
            if "num_epochs" in t: kwargs["num_epochs"] = int(t["num_epochs"])
            if "learning_rate" in t: kwargs["learning_rate"] = float(t["learning_rate"])
            if "weight_decay" in t: kwargs["weight_decay"] = float(t["weight_decay"])
            if "early_stopping_patience" in t: kwargs["early_stopping_patience"] = int(t["early_stopping_patience"])
            if "val_split_ratio" in t: kwargs["val_split_ratio"] = float(t["val_split_ratio"])
            if "seed" in t: kwargs["seed"] = int(t["seed"])

        if "loss" in data and isinstance(data["loss"], dict):
            l = data["loss"]
            if "alpha" in l: kwargs["loss_alpha"] = float(l["alpha"])
            if "beta" in l: kwargs["loss_beta"] = float(l["beta"])
            if "loss_alpha" in l: kwargs["loss_alpha"] = float(l["loss_alpha"])
            if "loss_beta" in l: kwargs["loss_beta"] = float(l["loss_beta"])

        if "evaluation" in data and isinstance(data["evaluation"], dict):
            e = data["evaluation"]
            if "target_directional_accuracy" in e: kwargs["target_directional_accuracy"] = float(e["target_directional_accuracy"])
            if "confidence_margin" in e: kwargs["confidence_margin"] = float(e["confidence_margin"])

        if "forecasting" in data and isinstance(data["forecasting"], dict):
            f = data["forecasting"]
            if "forecast_days" in f: kwargs["forecast_days"] = int(f["forecast_days"])
            if "confidence_level" in f: kwargs["confidence_level"] = float(f["confidence_level"])

        if "paths" in data and isinstance(data["paths"], dict):
            p = data["paths"]
            if "output_dir" in p: kwargs["output_dir"] = p["output_dir"]
            if "data_dir" in p: kwargs["data_dir"] = p["data_dir"]
            if "raw_data_path" in p: kwargs["raw_data_path"] = p["raw_data_path"]

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        for k, v in data.items():
            if k in valid_fields and v is not None:
                kwargs[k] = v
            elif k == "price_tickers":
                kwargs["_price_tickers"] = list(v)

        return cls(**kwargs)

    @classmethod
    def from_json(cls, json_path: Union[str, Path] = "config.json") -> "ForecasterConfig":
        p = Path(json_path)
        if not p.exists():
            alt_candidates = [
                Path.cwd() / json_path,
                Path(__file__).resolve().parent / json_path,
                Path(__file__).resolve().parent.parent / json_path,
            ]
            found = False
            for alt in alt_candidates:
                if alt.exists():
                    p = alt
                    found = True
                    break
            if not found:
                raise FileNotFoundError(
                    f"Configuration file '{json_path}' was not found in: "
                    f"{[str(p)] + [str(a) for a in alt_candidates]}"
                )

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"[CONFIG] Configuration loaded from: {p}")
        return cls.from_dict(data)
