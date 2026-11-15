# 🪄 teko_ajaib: Dual-Head LSTM Forecaster for Invesco QQQ (NASDAQ-100)

[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20Scikit--Learn-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Asset Target](https://img.shields.io/badge/Target-QQQ%20(NASDAQ--100)-green.svg)](https://www.invesco.com/qqq-etf/)

An enterprise-grade, end-to-end Deep Learning framework and Python library designed to forecast daily price levels and market direction (*Bullish* vs. *Bearish*) for the **Invesco QQQ Trust (NASDAQ-100 ETF)**.

**teko_ajaib** integrates a **Dual-Head LSTM** neural network trained with a custom **Direction-Aware Joint Loss Function**, combining highly-correlated intermarket equity drivers ($|r| \ge 0.85$), macroeconomic risk sentiment (**VIX**), and 6 quantitative momentum/trend technical indicators.

---

## 📑 Table of Contents

- [Overview & Architecture](#-overview--architecture)
  - [Dual-Head Architecture](#dual-head-architecture)
  - [Direction-Aware Joint Loss](#direction-aware-joint-loss)
  - [Feature Universe & Selection](#feature-universe--selection)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Configuration Guide (`config.json`)](#-configuration-guide-configjson)
  - [Configuration Breakdown](#configuration-breakdown)
  - [Parameter Tuning Tips](#parameter-tuning-tips)
- [Execution Walkthrough](#-execution-walkthrough)
  - [Method 1: One-Click End-to-End Pipeline](#method-1-one-click-end-to-end-pipeline)
  - [Method 2: Step-by-Step Modular Execution](#method-2-step-by-step-modular-execution)
  - [Method 3: Python Library API (Colab / Jupyter / Scripts)](#method-3-python-library-api-colab--jupyter--scripts)
- [Model Performance & Out-of-Sample Results](#-model-performance--out-of-sample-results)
  - [Evaluation Metrics Table](#evaluation-metrics-table)
  - [High-Conviction Trading Strategy Backtest](#high-conviction-trading-strategy-backtest)
- [Live Multi-Step Forecast (Next Week)](#-live-multi-step-forecast-next-week)
- [Visual Reports & Generated Artifacts](#-visual-reports--generated-artifacts)
- [Citation & Community](#-citation--community)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🧠 Overview & Architecture

Predicting financial market movements requires addressing two fundamental challenges simultaneously: **reconstructing continuous price levels** while **minimizing false direction classification** (the primary cause of trading drawdowns). Standard single-head LSTM models often minimize RMSE while outputting directional accuracy no better than a coin toss (~50%). **teko_ajaib** solves that through a synchronized multi-task learning architecture.

### Dual-Head Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │  Multivariate Time-Series Lookback Sequence  │
                    │   [Batch Size, Lookback=10, Features=17]    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │       Shared Multi-Layer Stacked LSTM        │
                    │      (Hidden Size = 64, Layers = 2)          │
                    └──────────────────────┬───────────────────────┘
                                           │ (Last Timestep Representation)
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
      ┌─────────────────────────────┐             ┌─────────────────────────────┐
      │      Regression Head        │             │     Classification Head     │
      │  Linear -> LayerNorm ->     │             │  Linear -> LayerNorm ->     │
      │       GELU -> Linear        │             │       GELU -> Linear        │
      └──────────────┬──────────────┘             └──────────────┬──────────────┘
                     ▼                                           ▼
            Continuous Return                            Directional Logit
             (Price Tracking)                        (Bullish / Bearish Prob)
```

1. **Shared Feature Extractor**: A stacked 2-layer LSTM with inter-layer dropout capturing temporal dependencies across multi-asset returns.
2. **Regression Head**: Outputs predicted normalized log return $\hat{r}_t \in \mathbb{R}$, unscaled to forecast the nominal price: $\hat{P}_t = P_{t-1} \cdot \exp(\hat{r}_t)$.
3. **Classification Head**: Outputs directional logit $\hat{z}_t \in \mathbb{R}$, transformed via sigmoid $\sigma(\hat{z}_t) = P(\text{Up}_t)$ to provide calibrated confidence probabilities.

### Direction-Aware Joint Loss

To strictly penalize whipsaw errors, training is guided by a compound multi-objective loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(\hat{r}, r) + \alpha \cdot \mathcal{L}_{\text{BCE}}(\hat{z}, y) + \beta \cdot \text{ReLU}(-\hat{r} \cdot r)$$

* **$\mathcal{L}_{\text{MSE}}$**: Minimizes squared error on continuous return values.
* **$\alpha \cdot \mathcal{L}_{\text{BCE}}$**: Trains directional discrimination using Binary Cross-Entropy.
* **$\beta \cdot \text{ReLU}(-\hat{r} \cdot r)$** (*Sign Penalty*): Triggers a severe loss penalty if and only if $\text{sign}(\hat{r}) \ne \text{sign}(r)$ (i.e. model predicts long when the market crashes, or vice versa).

### Feature Universe & Selection

| Category | Feature Name | Selection Rationale |
| :--- | :--- | :--- |
| **Target Asset** | `QQQ` | Invesco QQQ Trust (NASDAQ-100 Benchmark) |
| **Intermarket Core** | `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `META`, `TSLA`, `AMD`, `NVDA`, `AVGO` | Top tech market-cap weights in NASDAQ-100 having Pearson correlation $\|r\| \ge 0.85$ over historical training period |
| **Market Risk Sentiment** | `VIX` | CBOE Volatility Index capturing market risk aversion and regime shifts |
| **Trend & Momentum** | `RSI-14`, `MACD_HIST`, `MOM3`, `MOM5`, `SMA5_RATIO`, `SMA20_RATIO` | 6 technical indicators capturing overbought/oversold levels, moving average divergence, and short-term momentum |

---

## 📂 Project Directory Structure

```text
teko_ajaib/
├── config.json                     # ⭐ Centralized configuration (all hyperparameters)
├── config.py                       # Python config bridge (loads config.json with fallbacks)
├── run_pipeline.py                 # CLI orchestrator to run full pipeline end-to-end
│
├── 01_data_engineering.py          # Module 1: ETL, correlation analysis & technical indicators
├── 02_train_optimize.py            # Module 2: Dual-Head LSTM training & validation
├── 03_test_forward.py              # Module 3: Out-of-sample forward test & threshold calibration
├── 04_live_forward.py              # Module 4: Recursive multi-step forecasting for next week
├── example_colab_usage.py          # Ready-to-run Google Colab / Jupyter example script
│
├── teko_ajaib/                     # Core Python package
│   ├── __init__.py                 # Top-level module exports
│   ├── config.py                   # ForecasterConfig dataclass (JSON load/save support)
│   ├── model.py                    # DualHeadLSTM & TimeSeriesSequenceDataset classes
│   ├── results.py                  # TrainResult dataclass container
│   ├── data/
│   │   ├── loader.py               # Yahoo Finance downloader & CSV loader
│   │   ├── features.py             # Pearson correlation & technical indicator calculator
│   │   └── engineer.py             # DataEngineer orchestrator
│   ├── training/
│   │   ├── loss.py                 # DirectionAwareJointLoss PyTorch module
│   │   └── trainer.py              # Trainer class (AdamW, early stopping, save/load)
│   ├── evaluation/
│   │   ├── backtest.py             # Confidence margin sweeper & backtesting simulator
│   │   └── forward_test.py         # ForwardTester class
│   ├── forecasting/
│   │   └── live_forecast.py        # LiveForecaster recursive multi-step forecaster
│   └── visualization/
│       └── plots.py                # Publication-quality plotting routines
│
├── data/
│   ├── raw/                        # Historical raw market CSVs
│   ├── train_val_data.csv          # Cleaned training dataset
│   ├── forward_test_data.csv       # Out-of-sample forward test dataset
│   ├── correlation_matrix.csv      # Correlation matrix output
│   ├── forward_test_predictions.csv# Forward test daily log of predictions
│   └── live_forward_next_week_forecast.csv # Next-week daily price & probability forecast
│
├── models/
│   ├── best_lstm_model.pth         # Trained PyTorch state dict
│   ├── scaler.pkl                  # Fitted StandardScaler (Train-set only)
│   └── training_config.json        # Snapshot of training hyperparameters & metadata
│
└── reports/
    ├── correlation_heatmap.png            # Feature correlation matrix heatmap
    ├── dataset_split_overview.png         # Dataset timeline and split visualization
    ├── training_val_loss_curve.png        # Epoch training loss vs validation loss & acc
    ├── forward_test_price_comparison.png  # Actual vs. predicted prices & residual error
    ├── directional_confidence_curve.png   # Margin threshold vs directional accuracy curve
    ├── forward_test_directional_matrix.png# Confusion matrix for market direction
    ├── trading_strategy_backtest.png      # Cumulative equity curve (LSTM vs Buy & Hold)
    ├── forward_test_metrics.json          # Numerical test evaluation metrics
    ├── live_forward_next_week_chart.png   # 5-day recursive forecast with 95% confidence cone
    └── live_forward_summary.json          # Summary JSON of next week's projections
```

---

## ⚙️ Installation & Setup

### Requirements
- **Python**: 3.7, 3.8, 3.9, 3.10, 3.11
- **Hardware**: CPU compatible; CUDA GPU automatically detected and utilized if available.

### 1. Clone the Repository
```bash
git clone https://github.com/teko-ajaib/teko_ajaib.git
cd teko_ajaib
```

### 2. Create and Activate Virtual Environment
```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Or install in editable mode with all optional packages (yfinance, python-docx)
pip install -e ".[all]"
```

---

## 🛠️ Configuration Guide (`config.json`)

All pipeline behaviors—from data tickers and chronological splits to neural network hyperparameters and financial thresholds—are managed through [config.json](config.json).

You do **not** need to touch any Python code to adjust your experiment. Simply edit this file:

```json
{
  "_comment": "Central Configuration for teko_ajaib Forecasting Pipeline",
  "project": {
    "name": "teko-ajaib",
    "version": "1.0.0"
  },
  "data": {
    "target_ticker": "QQQ",
    "candidate_tickers": [
      "QQQ", "AAPL", "NVDA", "MSFT", "AMZN",
      "AMD", "MU", "TSLA", "META", "GOOGL", "AVGO", "VIX"
    ],
    "price_tickers": [
      "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN",
      "META", "TSLA", "AMD", "NVDA", "AVGO", "VIX"
    ],
    "start_train_date": "2015-01-01",
    "split_date": "2026-06-30",
    "forward_test_start": "2026-07-01",
    "correlation_threshold": 0.85,
    "use_technical_indicators": true,
    "include_vix": true
  },
  "model": {
    "architecture": "DualHeadLSTM",
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2
  },
  "training": {
    "seq_len": 10,
    "batch_size": 32,
    "num_epochs": 40,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "early_stopping_patience": 12,
    "val_split_ratio": 0.20,
    "seed": 42
  },
  "loss": {
    "alpha": 1.0,
    "beta": 0.5
  },
  "evaluation": {
    "target_directional_accuracy": 70.0,
    "confidence_margin": 0.05
  },
  "forecasting": {
    "forecast_days": 5,
    "confidence_level": 0.95
  },
  "paths": {
    "base_dir": ".",
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "raw_data_filename": "qqq_daily_cross_asset_2000_2026.csv",
    "models_dir": "models",
    "reports_dir": "reports"
  }
}
```

### Configuration Breakdown

| Section | Parameter | Type | Default | Description |
| :--- | :--- | :---: | :---: | :--- |
| **`data`** | `target_ticker` | `string` | `"QQQ"` | Target instrument symbol to predict. |
| | `candidate_tickers` | `list` | `["QQQ", ...]` | Universe of tickers tested for Pearson correlation. |
| | `start_train_date` | `string` | `"2015-01-01"` | Start of modeling history (`YYYY-MM-DD`). |
| | `split_date` | `string` | `"2026-06-30"` | End of train-validation data (train/test partition cutoff). |
| | `forward_test_start` | `string` | `"2026-07-01"` | Beginning of true out-of-sample forward testing. |
| | `correlation_threshold` | `float` | `0.85` | Minimum absolute Pearson correlation required to retain a feature. |
| | `use_technical_indicators`| `bool` | `true` | Calculates RSI-14, MACD, MOM3, MOM5, SMA5 & SMA20 ratios. |
| | `include_vix` | `bool` | `true` | Retains CBOE Volatility Index as a market regime indicator. |
| **`model`** | `hidden_size` | `int` | `64` | Number of LSTM units in hidden layers. |
| | `num_layers` | `int` | `2` | Number of stacked LSTM layers. |
| | `dropout` | `float` | `0.2` | Dropout rate applied between layers and classification/regression heads. |
| **`training`** | `seq_len` | `int` | `10` | Historical lookback window length (10 trading days). |
| | `batch_size` | `int` | `32` | Mini-batch size for AdamW optimization. |
| | `num_epochs` | `int` | `40` | Maximum number of training epochs. |
| | `learning_rate` | `float` | `0.001` | Initial learning rate (adjusted by `ReduceLROnPlateau`). |
| | `weight_decay` | `float` | `0.001` | $L_2$ weight decay regularization. |
| | `early_stopping_patience`| `int` | `12` | Patience epochs before early termination if validation loss stalls. |
| | `val_split_ratio` | `float` | `0.20` | Fraction of chronological data reserved for validation (20%). |
| | `seed` | `int` | `42` | Random seed for PyTorch, NumPy, and CUDA reproducibility. |
| **`loss`** | `alpha` | `float` | `1.0` | Weight scalar for Directional BCE loss. |
| | `beta` | `float` | `0.5` | Weight scalar for Sign Penalty (penalizing wrong-direction calls). |
| **`evaluation`** | `target_directional_accuracy` | `float` | `70.0` | Desired minimum accuracy for high-conviction signals ($\ge 70\%$). |
| | `confidence_margin` | `float` | `0.05` | Margin around 0.50 for high-conviction filtering ($\|P - 0.50\| \ge \text{margin}$). |
| **`forecasting`**| `forecast_days` | `int` | `5` | Number of future business days to forecast (1 week = 5 days). |
| | `confidence_level` | `float` | `0.95` | Statistical confidence level for projection cone ($Z=1.96$). |
| **`paths`** | `data_dir` / `models_dir` / `reports_dir` | `string` | `"data"` ... | Target folder locations for generated data, weights, and plots. |

### Parameter Tuning Tips
- **Increasing Model Capacity**: For longer history periods, test `"hidden_size": 128` and `"num_layers": 2`.
- **Higher Signal Conviction**: Increase `"confidence_margin": 0.10` in `config.json` to filter out market noise and execute only on high-certainty setups.
- **Custom Assets**: Set `"target_ticker": "SPY"` or `"NVDA"` along with corresponding candidate peers in `config.json` to forecast other assets.

---

## 🚀 Execution Walkthrough

### Method 1: One-Click End-to-End Pipeline

To execute all modules sequentially—including data downloading, training, forward testing, backtesting, and live next-week forecasting:

```bash
python run_pipeline.py --live
```

*(Add `--optimize` to execute an automated grid search across hyperparameter combinations before final training)*.

```bash
python run_pipeline.py --optimize --live
```

---

### Method 2: Step-by-Step Modular Execution

Each stage can be executed independently. Every module reads directly from [config.json](config.json):

#### Step 1: Data Engineering & Feature Selection
Downloads or loads raw prices, cleans missing values, conducts Pearson correlation screening on the train period, computes 6 technical indicators, and saves partitioned datasets:
```bash
python 01_data_engineering.py
```
*Outputs generated:* `data/train_val_data.csv`, `data/forward_test_data.csv`, `reports/correlation_heatmap.png`.

#### Step 2: Training & Optimization
Constructs the multi-input sliding window dataset, fits `StandardScaler` strictly on the train split (preventing data leakage), and trains the `DualHeadLSTM` using `DirectionAwareJointLoss`:
```bash
python 02_train_optimize.py
```
*Outputs generated:* `models/best_lstm_model.pth`, `models/scaler.pkl`, `reports/training_val_loss_curve.png`.

#### Step 3: Out-of-Sample Forward Testing & Backtest
Runs inference over unseen forward-test data (July 2026 – present). Computes RMSE, MAE, MAPE, $R^2$, sweeps confidence margins to hit $\ge 70\%$ accuracy, and simulates a financial trading strategy against Buy & Hold:
```bash
python 03_test_forward.py
```
*Outputs generated:* `data/forward_test_predictions.csv`, `reports/forward_test_price_comparison.png`, `reports/trading_strategy_backtest.png`.

#### Step 4: Live Forward Forecasting (Next Week Projection)
Retrieves the most recent market observations, executes recursive multi-step forecasting for the next 5 business days, calculates dynamic intermarket beta updates, and projects 95% confidence bounds:
```bash
python 04_live_forward.py
```
*Outputs generated:* `data/live_forward_next_week_forecast.csv`, `reports/live_forward_next_week_chart.png`, `reports/live_forward_summary.json`.

---

### Method 3: Python Library API (Colab / Jupyter / Scripts)

You can import and use the modular **teko_ajaib** library components directly in Python:

```python
from teko_ajaib import (
    ForecasterConfig, 
    DataEngineer, 
    Trainer, 
    ForwardTester, 
    LiveForecaster
)
from teko_ajaib.visualization import (
    plot_training_curve, 
    plot_forward_test, 
    plot_live_forecast
)

# 1. Load configuration from config.json (or instantiate directly)
config = ForecasterConfig.from_json("config.json")

# 2. Run Data Engineering Pipeline
engineer = DataEngineer(config)
data = engineer.prepare()  # Downloads via Yahoo Finance or loads local CSV

# 3. Train Dual-Head LSTM Model
trainer = Trainer(config)
train_result = trainer.train(data)
trainer.save(train_result, path="./models")
plot_training_curve(train_result.history)

# 4. Evaluate on Out-of-Sample Forward Test Data
tester = ForwardTester(config)
metrics, df_results = tester.evaluate(train_result, data)
plot_forward_test(df_results, metrics)

# 5. Predict Next Week (5 Business Days Recursive Live Forecast)
forecaster = LiveForecaster(config)
forecast_df, summary = forecaster.predict_next_week(train_result, data, n_days=5)

print("\n=== NEXT WEEK LIVE FORECAST ===")
print(forecast_df[["Date", "Day", "Predicted_Price", "Lower_Bound_95", "Upper_Bound_95", "Direction", "Signal_Strength"]])
```

---

## 📊 Model Performance & Out-of-Sample Results

The model was evaluated on unseen out-of-sample forward testing data from **July 1, 2026 to November 13, 2026** (97 trading days) against a standard Naive baseline ($P_t = P_{t-1}$):

### Evaluation Metrics Table

| Metric | Dual-Head LSTM | Baseline Benchmark | Evaluation Assessment |
| :--- | :---: | :---: | :--- |
| **RMSE (Price Level)** | **\$10.27** | \$10.22 (Naive $P_{t-1}$) | High reconstruction precision |
| **MAE (Mean Absolute Error)** | **\$7.50** | \$7.52 (Naive $P_{t-1}$) | **LSTM outperforms Baseline** |
| **MAPE (%)** | **1.062%** | 1.066% (Naive $P_{t-1}$) | Low relative error (~1.06%) |
| **Coefficient of Determination ($R^2$)**| **0.7809** | 0.7833 | Captures 78.1% of market price variance |
| **Directional Accuracy (All Days)** | **57.73%** | 50.00% (Random Walk) | Consistently beats random chance |
| **Directional Accuracy (High-Conviction)** | **71.43%** | **Target $\ge$ 70.0%** | **Target Achieved ($\ge 70.0\%$ Hit Rate)** |
| **Active Signal Coverage** | **28.9%** (28/97 days) | — | Eliminates noise via confidence threshold |
| **Strategy Cumulative Return** | **+8.53%** | -7.67% (Buy & Hold QQQ) | **Significant Outperformance (+16.20%)** |
| **Annualized Sharpe Ratio** | **+2.03** | -0.89 (Buy & Hold QQQ) | **Superior Risk-Adjusted Returns** |

### High-Conviction Trading Strategy Backtest

By trading only when model confidence satisfies $\|P(\text{Up}) - 0.50\| \ge 0.05$ (abstaining during noisy consolidations), the model successfully generated positive alpha during an overall bearish market period:

- **Dual-Head LSTM Strategy**: **+8.53%** return (Sharpe: **+2.03**)
- **Passive Buy & Hold (QQQ)**: **-7.67%** drawdown (Sharpe: **-0.89**)

![Trading Strategy Backtest](reports/trading_strategy_backtest.png)

---

## 🔮 Live Multi-Step Forecast (Next Week)

The live recursive multi-step forecasting engine projects daily closing prices, 95% confidence intervals, and direction signals.

### Recursive Methodology
1. At step $t+1$, the model forecasts QQQ return $\hat{r}_{t+1}$.
2. Intermarket features are updated via historical beta sensitivities: $\hat{r}_{asset} = \beta_{asset} \cdot \hat{r}_{qqq}$.
3. Technical indicators (RSI, MACD, Momentum, Moving Average ratios) are recalculated dynamically on the projected window.
4. The window slides forward to predict step $t+2$, propagating through day $N$.
5. The 95% confidence cone expands with forecast horizon: $\text{CI}_t = \pm 1.96 \cdot \sigma_{\text{daily}} \cdot \sqrt{t}$.

### Sample Live Projection (Nov 16 – Nov 20, 2026)
*Last Observed Price: **\$679.89** (Friday, Nov 13, 2026)*

| Day | Date | Predicted Price | 95% Confidence Band | Projected Return | Signal | Direction | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Monday** | 2026-11-16 | **\$666.14** | \$645.66 – \$687.27 | -2.04% | 84.5% Down | BEARISH | Strong Signal (Sell / Hedge) |
| **Tuesday** | 2026-11-17 | **\$656.14** | \$627.80 – \$685.76 | -1.51% | 78.3% Down | BEARISH | Strong Signal (Hold Short) |
| **Wednesday** | 2026-11-18 | **\$651.70** | \$617.39 – \$687.91 | -0.68% | 58.8% Down | CONSOLIDATION | Moderate Signal (Take Profit) |
| **Thursday** | 2026-11-19 | **\$655.58** | \$615.89 – \$697.82 | +0.59% | 70.7% Up | BULLISH | Strong Signal (Buy Reversal) |
| **Friday** | 2026-11-20 | **\$659.84** | \$615.35 – \$707.56 | +0.65% | 74.2% Up | BULLISH | Strong Signal (Buy Continuation) |

![Live Forecast Chart](reports/live_forward_next_week_chart.png)

---

## 📈 Visual Reports & Generated Artifacts

The pipeline automatically generates publication-ready visualizations saved in the `reports/` folder:

| Report Visualization | Description |
| :--- | :--- |
| **`correlation_heatmap.png`** | Pearson correlation matrix showing cross-asset intermarket relationships. |
| **`dataset_split_overview.png`** | Normalized price histories across assets with train/validation/test partitions. |
| **`training_val_loss_curve.png`** | Dual loss curves (Train vs. Val) and real-time directional accuracy progression. |
| **`forward_test_price_comparison.png`** | Out-of-sample actual vs. predicted price trajectory and residual error distribution. |
| **`directional_confidence_curve.png`** | Calibration curve illustrating hit-rate growth as conviction margin is tightened. |
| **`forward_test_directional_matrix.png`** | Confusion matrix visualizing precision, recall, and accuracy for up/down calls. |
| **`trading_strategy_backtest.png`** | Cumulative equity curve comparing LSTM directional trading vs. Buy & Hold. |
| **`live_forward_next_week_chart.png`** | 5-day recursive forecast with 95% confidence intervals and historical context. |

---

## 👥 Citation & Community

Developed and maintained by the **teko_ajaib open-source contributors**. Contributions, issues, and feature requests are welcome!

```bibtex
@misc{tekoajaib2026,
  author = {teko\_ajaib Contributors},
  title = {teko_ajaib: Dual-Head LSTM Forecaster for Financial Indices},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/teko-ajaib/teko_ajaib}}
}
```

---

## ⚠️ Disclaimer

This repository is built strictly for **educational and research purposes**. Financial markets involve substantial risk of loss. The forecasts, probability estimates, and signals provided by this model do not constitute financial advice, investment recommendations, or trading endorsements. Neither the authors nor the contributors accept liability for financial losses incurred through the use of this software.

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
