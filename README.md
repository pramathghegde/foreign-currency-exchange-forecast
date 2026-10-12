# Foreign Currency Exchange Rate Forecast

A Machine Learning project replicating and extending the Stanford CS229 (Spring 2019) research paper: **"Foreign Currency Exchange Rate Forecast"** by *Jiahao Zhang, Jialu Sun, and Jiequan Zhang*.

---

## 1. Project Objectives & Overview

Foreign exchange (FX) markets are among the largest and most volatile financial markets globally. Traditional econometric structural models often fail to outperform a simple random walk benchmark in out-of-sample forecasting (Meese & Rogoff, 1983). 

This project explores whether modern supervised machine learning techniques can:
1. Predict the directional change (sign prediction: Up vs. Down) of daily exchange rates.
2. Forecast next-day log-return values (regression).
3. Generate economically profitable trading strategies through algorithmic simulation.
4. Benchmark an additional non-linear ensemble algorithm (**XGBoost**) not included in the original 2019 study.

---

## 2. Reference Paper Scope & Extension Requirements

### Target Currency Pairs (Expressed Relative to USD)
- **GBP/USD** (U.S. Dollar to British Pound)
- **USD/EUR** (U.S. Dollar to Euro)
- **USD/JPY** (U.S. Dollar to Japanese Yen)

### Dataset Timeline & Chronological Splits
- **Total Historical Window:** January 2001 – December 2018 (Daily closing prices).
- **Partition Ratio:** 80% Train, 10% Validation, 10% Test (Strict chronological ordering; no future-leakage).
- **Test Set Window:**
  - EUR/USD: March 3, 2017 – November 30, 2018.
  - GBP/USD & USD/JPY: March 7, 2017 – November 30, 2018.

### Models
| Model Type | Reference Models (Paper) | Extension Model |
| :--- | :--- | :--- |
| **Classification** (Sign) | Logistic Regression, Lasso, Ridge, SVM (RBF), Random Forest, MLP | **XGBoost Classifier** |
| **Regression** (Log Return) | Lasso, Ridge, SVM (RBF & Poly), Random Forest, MLP | **XGBoost Regressor** |
| **Trading Simulation** | Random Forest (Best performer in paper) | **XGBoost vs. RF vs. Baselines** |

---

## 3. Planned Experiments

### Experiment 1: Sign Prediction (Binary Classification)
- **Target:** $y_t = \mathbb{I}(r_t > 0)$, where $r_t = \ln(P_t / P_{t-1})$.
- **Evaluation Metrics:**
  - Accuracy (Acc)
  - Normalized Mean Squared Error (NMSE)
  - Mean Absolute Error (MAE)
  - Directional Symmetry (DS)
  - Correct Up Trend (CU)
  - Correct Down Trend (CD)

### Experiment 2: Value Prediction (Regression)
- **Target:** Daily log return value $r_{t+1}$.
- **Evaluation Metrics:**
  - Coefficient of Determination ($R^2 = 1 - u/v$)
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)

### Experiment 3: Trading Strategy Backtest Simulation
- **Initial Capital:** $100 USD at the start of the test window.
- **Decision Rule:**
  - Compute predicted ratio $\hat{R}_{t} = \frac{\widehat{P}_{t+1}}{P_t}$.
  - If all predicted ratios $\ge 1$, hold USD cash.
  - If any predicted ratio $< 1$ (USD expected to depreciate against foreign currency), convert 100% of capital into the foreign currency with the lowest ratio for 1 day, then convert back to USD at end of day.
- **Portfolios Evaluated:**
  - Single currency: GBP, JPY, EUR
  - Two-currency combinations: GBP+JPY, JPY+EUR, GBP+EUR
  - Combined portfolio: GBP+JPY+EUR

---

## 4. Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14.6)
- Git

### Installation
Clone the repository and install required packages:
```bash
# Clone repository
git clone <repo-url>
cd "ML Mini Project"

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependency Overview
- `numpy`, `pandas`: Vectorized calculations and time-series dataframe manipulation.
- `scikit-learn`: Linear models, SVM, Random Forest, MLP, scaling, and evaluation metrics.
- `xgboost`: Gradient boosting extension model.
- `matplotlib`, `seaborn`: Visualization and performance plotting.
- `requests`, `yfinance`: Automated data ingestion from Yahoo Finance and FRED.

---

## 5. Repository Structure

```text
├── data/
│   ├── raw/                 # Downloaded raw historical data (.gitkeep)
│   ├── processed/           # Aligned features and cleaned series (.gitkeep)
│   └── DATA_SOURCES.md      # Detailed source documentation and ticker mapping
├── notebooks/               # Interactive exploration and prototyping notebooks (.gitkeep)
├── reports/                 # Project documentation and milestone reports (.gitkeep)
├── results/
│   ├── figures/             # Exported performance curves and cash plots (.gitkeep)
│   └── tables/              # Exported comparative metric tables (CSV/LaTeX) (.gitkeep)
├── src/
│   └── __init__.py          # Modular Python source package
├── .gitignore               # Excludes caches, venvs, and generated outputs
├── requirements.txt         # Minimal project dependencies
└── README.md                # Project documentation and reproduction guide
```

---

## 6. Execution Workflow (Future Pipeline)

Once data acquisition and feature engineering modules are implemented, the execution sequence will be:

```bash
# 1. Download market indices, FX rates, and macroeconomic indicators
python src/data_loader.py

# 2. Build 15 technical and macro features with chronological train/val/test splits
python src/feature_engineering.py

# 3. Train models and generate classification benchmarks (Table 1 replication)
python src/train_classification.py

# 4. Train models and generate regression benchmarks (Table 2 replication)
python src/train_regression.py

# 5. Run trading strategy backtester and produce cash growth curves (Table 3 & Figure 3 replication)
python src/backtest_simulation.py
```
