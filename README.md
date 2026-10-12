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
- **GBP/USD** (U.S. Dollar to British Pound) — FRED `DEXUSUK`
- **EUR/USD** (U.S. Dollar to Euro) — FRED `DEXUSEU`
- **USD/JPY** (U.S. Dollar to Japanese Yen) — FRED `DEXJPUS`

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

## 3. Environment Setup & Dependency Installation

### Step 1: Create Virtual Environment
A local virtual environment `.venv` configured with system site-package access (to comply with Windows Application Control policies regarding compiled extensions):

```powershell
# In PowerShell:
py -m venv .venv --system-site-packages
```

### Step 2: Install Project Dependencies
Install `yfinance` and `xgboost` inside `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Verified Packages in `.venv`
- `numpy` 2.5.1
- `pandas` 3.0.5
- `scikit-learn` 1.9.0
- `xgboost` 3.4.1
- `yfinance` 1.7.0
- `requests` 2.34.2
- `curl_cffi` 0.16.3 (enables secure, throttled-free FRED downloads)
- `matplotlib` 3.11.1
- `seaborn` 0.13.2

---

## 4. Historical Data Ingestion & Caching

Download and cache raw daily data from January 1, 2001 through December 31, 2018 into `data/raw/`:

```powershell
# Run the cached downloader:
.\.venv\Scripts\python.exe src/data_loader.py

# To force re-download and bypass existing raw cache:
.\.venv\Scripts\python.exe src/data_loader.py --force
```

### Data Pipeline Details
- **Exchange Rates (FRED):** `DEXUSUK` (GBP/USD), `DEXUSEU` (EUR/USD), `DEXJPUS` (USD/JPY).
- **Market Indices (Yahoo Finance):** `^GSPC` (S&P 500), `^OEX` (S&P 100), `^IXIC` (NASDAQ), `^DJI` (Dow Jones).
- **Macroeconomic Indicators (FRED):** `DFF` (Fed Funds Rate), `CPIAUCNS` (CPI), `USREC` (Recession Indicator), `GFDEBTN` (Public Debt).

---

## 5. Raw Data Validation

Inspect and validate downloaded datasets to verify date coverage, duplicate avoidance, and absence of invalid/negative rates:

```powershell
.\.venv\Scripts\python.exe src/validate_data.py
```

Expected output: **11 / 11 files PASSED**.

---

## 6. Feature Engineering & Dataset Preprocessing

Run the automated feature generation pipeline to construct technical indicators, align macroeconomic factors with conservative publication lags, generate chronological train/val/test splits, and export standard-scaled datasets:

```powershell
# Run the feature engineering pipeline:
.\.venv\Scripts\python.exe src/feature_engineering.py
```

### Feature Set Overview (23 Total Features)
- **15 Technical Indicators (Paper Section 3.2):**
  - Current & previous log return: $r_t, r_{t-1}$
  - Moving averages of log returns: 5, 10, 20, 60, 120 days
  - Log price polynomials: $\ln(P_t), (\ln(P_t))^2, (\ln(P_t))^3$
  - Moving averages of log price: 5, 10, 20, 60, 220 days
- **8 Financial & Macroeconomic Indicators (Paper Section 3.1):**
  - Daily log returns: S&P 500 (`^GSPC`), S&P 100 (`^OEX`), NASDAQ (`^IXIC`), DJIA (`^DJI`)
  - Policy & macro levels: Fed Funds Effective Rate (`DFF`)
  - Monthly CPI rate of change with 15-day publication lag (`CPIAUCNS`)
  - Monthly NBER recession indicator with 30-day reporting lag (`USREC`)
  - Quarterly public debt growth with 45-day reporting lag (`GFDEBTN`)

---

## 7. Processed Dataset Validation

Inspect the engineered datasets in `data/processed/` for NaNs, infinites, monotonic chronological order, target shift correctness, and absence of train/val/test overlap:

```powershell
.\.venv\Scripts\python.exe src/validate_features.py
```

Expected output: **ALL PROCESSED DATASETS PASSED VALIDATION WITH ZERO ERRORS**.

---

## 8. Experiment 1: Sign Prediction (Classification)

Train and evaluate the six reference classification models across GBP, EUR, and JPY:
1. **Logistic Regression** ($C$ tuned via validation)
2. **Lasso** ($L1$-penalized classification via `liblinear`)
3. **Ridge** ($L2$-penalized linear classification via `RidgeClassifier`)
4. **Support Vector Machine** (RBF kernel, $C$ and $\gamma$ tuned via validation)
5. **Random Forest** (1,000 trees, max depth 100 per paper specification)
6. **Multi-Layer Perceptron** (2 hidden layers per paper specification)

```powershell
# Run Experiment 1 classification pipeline:
.\.venv\Scripts\python.exe src/classification.py

# Validate classification outputs, metric integrity, and absence of target leakage:
.\.venv\Scripts\python.exe src/validate_classification.py
```

### Generated Classification Results
- **Validation Table:** `results/tables/classification_validation_results.csv`
- **Test Table:** `results/tables/classification_test_results.csv`
- **Paper Comparison Table:** `results/tables/classification_paper_comparison.csv`
- **Accuracy Comparison Bar Chart:** `results/figures/classification_accuracy_comparison.png`
- **Per-Currency Accuracy Plot:** `results/figures/classification_per_currency_accuracy.png`

---

## 9. Experiment 2: Numerical Return Regression

Train and evaluate the six reference regression models across GBP, EUR, and JPY to predict next-day continuous log return $r_{t+1}$:
1. **Random Forest Regressor** (1,000 trees, max depth 100 on unscaled features per paper specification)
2. **Multi-Layer Perceptron Regressor** (2 hidden layers, validation-tuned architecture and $L2$ penalty)
3. **Support Vector Regression (SVR with RBF kernel)** ($C$ and $\epsilon$ tuned via validation)
4. **Support Vector Regression (SVR with Polynomial kernel)** (Degree 2/3, $C$ and $\epsilon$ tuned via validation)
5. **Lasso Regression** ($\alpha$ tuned via validation)
6. **Ridge Regression** ($\alpha$ tuned via validation)

```powershell
# Run Experiment 2 regression pipeline:
.\.venv\Scripts\python.exe src/regression.py

# Validate regression outputs, target isolation, and finite metric integrity:
.\.venv\Scripts\python.exe src/validate_regression.py
```

### Generated Regression Results
- **Validation Table:** `results/tables/regression_validation_results.csv`
- **Test Table:** `results/tables/regression_test_results.csv`
- **Paper Table 2 Comparison Table:** `results/tables/regression_paper_comparison.csv`
- **Paper Table 4 RF Comparison Table:** `results/tables/regression_table4_rf_comparison.csv`
- **Actual vs. Predicted Return Plots:** `results/figures/regression_predictions_{gbp,eur,jpy}.png`
- **Model $R^2$ Comparison Bar Chart:** `results/figures/regression_r2_comparison.png`

---

## 10. Experiment 3: Trading Strategy Simulation & Portfolio Backtesting

Simulate the paper's quantitative currency portfolio trading strategy (Section 5.3, Table 3, Figure 3) starting with an initial capital of $100.00 across the out-of-sample test period (2017-03-07 to 2018-11-30, 438 trading days).

### Strategy Execution Rules & Conventions
- **Model:** Random Forest classifier (1,000 trees, max depth 100) trained strictly on 2001–2015 data.
- **Directional Signals:** At day $t$, the model predicts $\hat{y}_{t+1} \in \{0, 1\}$.
- **Single-Currency Positions:**
  - **GBP/USD & EUR/USD (Direct Quotes):** $\hat{y} = 1$ signals foreign currency appreciation; holding foreign currency realizes return factor $P_{t+1} / P_t$. When $\hat{y} = 0$, holding USD cash realizes return factor $1.00$.
  - **USD/JPY (Indirect Quote, `DEXJPUS`):** The raw quote is JPY per USD. The default `quote_long` backtest applies $P_{t+1} / P_t$ when $\hat{y} = 1$, representing a long USD / short JPY position (rather than holding JPY).
  - *Methodological Note:* Because the original paper does not document whether its JPY rate was inverted or detail its exact return formula, this represents a documented interpretation rather than verified exact replication.
- **Multi-Currency Combinations:**
  - Daily capital is divided equally among currencies signalling long ($\hat{y} = 1$).
  - If no currencies signal long, capital remains 100% in USD cash.

```powershell
# Run Stage 6 trading simulation backtest:
.\.venv\Scripts\python.exe src/trading_simulation.py

# Validate trading simulation outputs, capital accounting, and math integrity:
.\.venv\Scripts\python.exe src/validate_trading_simulation.py
```

### Generated Trading Simulation Results
- **Full Results Table:** `results/tables/trading_simulation_results.csv`
- **Paper Table 3 Comparison Table:** `results/tables/trading_paper_comparison.csv`
- **Figure 3 Reproduction Plot:** `results/figures/trading_figure3_portfolio_cash.png`
- **Final Balance Comparison Bar Chart:** `results/figures/trading_final_balance_comparison.png`

---

## 11. Stage 7: XGBoost Extension

Implements the novel proposed machine learning model (**XGBoost**) beyond the six reference models explored in the Stanford CS229 paper:
1. **XGBoost Classifier:** Predicts next-day direction sign (`target_clf`).
2. **XGBoost Regressor:** Predicts next-day continuous log return (`target_reg`).

### Training & Tuning Methodology
- **Validation Tuning:** Evaluated across depth $\{2, 3, 5\}$, learning rate $\{0.01, 0.03, 0.05, 0.10\}$, subsampling, and $L1/L2$ regularization ($\alpha, \lambda$).
- **Early Stopping & Leakage Isolation:** Models fit on training data with early stopping on the chronological validation set. The test set remains untouched until final scoring.
- **Fair Benchmark Comparison:** Evaluated on the identical 23 features, splits, and metrics alongside the reference models.

```powershell
# Run Stage 7 XGBoost experiments:
.\.venv\Scripts\python.exe src/xgboost_models.py

# Validate XGBoost outputs, split separation, and metric integrity:
.\.venv\Scripts\python.exe src/validate_xgboost.py
```

### Generated XGBoost Results
- **Validation Tables:** `results/tables/xgboost_classification_validation.csv`, `results/tables/xgboost_regression_validation.csv`
- **Test Tables:** `results/tables/xgboost_classification_test.csv`, `results/tables/xgboost_regression_test.csv`
- **Cross-Model Comparison Table:** `results/tables/xgboost_paper_comparison.csv`
- **Visualizations:**
  * `results/figures/xgboost_vs_reference_classification.png` (Accuracy comparison across models & currencies)
  * `results/figures/xgboost_vs_reference_regression.png` (Regression $R^2$ comparison across models & currencies)
  * `results/figures/xgboost_feature_importance.png` (XGBoost feature importance ranking)

---

## 12. Repository Structure

```text
├── data/
│   ├── raw/                 # Downloaded raw historical datasets (11 series + metadata.json)
│   ├── processed/           # Processed & scaled split datasets (21 CSV files + dataset_metadata.json)
│   └── DATA_SOURCES.md      # Detailed source documentation and ticker mapping
├── notebooks/               # Interactive exploration and prototyping notebooks (.gitkeep)
├── report/                  # Final comprehensive project report
│   └── final_project_report.md # Comprehensive academic reproduction and extension report
├── results/
│   ├── figures/             # Exported performance curves, cash plots, and comparison charts
│   └── tables/              # Exported comparative metric tables (CSV)
├── src/
│   ├── __init__.py          # Source package initializer
│   ├── data_loader.py       # Automated data downloader with local file caching
│   ├── validate_data.py     # Raw data validation and integrity checking suite
│   ├── feature_engineering.py # End-to-end feature engineering and chronological partitioner
│   ├── validate_features.py # Feature matrix integrity and leakage test suite
│   ├── classification.py    # Experiment 1 sign prediction training & evaluation suite
│   ├── validate_classification.py # Classification validation and benchmark test suite
│   ├── regression.py        # Experiment 2 return regression training & evaluation suite
│   ├── validate_regression.py # Regression validation and benchmark test suite
│   ├── trading_simulation.py # Experiment 3 trading strategy backtester & portfolio simulator
│   ├── validate_trading_simulation.py # Trading simulation validation test suite
│   ├── xgboost_models.py    # Stage 7 XGBoost classifier & regressor extension suite
│   └── validate_xgboost.py  # XGBoost validation test suite
├── .gitignore               # Excludes caches, venvs, and generated outputs
├── requirements.txt         # Minimal project dependencies
└── README.md                # Project documentation and reproduction guide
```

---

## 13. Project Report & Complete Validation

The complete, rigorous project report is available at:
👉 **[Final Project Report](report/final_project_report.md)**

It provides comprehensive theoretical background, dataset audit, classification/regression comparisons, the mathematical root-cause diagnosis of the Random Forest $R^2$ discrepancy, trading curve analyses, and XGBoost benchmark results.

### Complete Project Validation Suite
To execute the automated validation suite across all 6 modules (verifying data, features, classification, regression, trading simulation, and XGBoost):

```powershell
.\.venv\Scripts\python.exe -c "
import subprocess, sys
scripts = [
    'src/validate_data.py',
    'src/validate_features.py',
    'src/validate_classification.py',
    'src/validate_regression.py',
    'src/validate_trading_simulation.py',
    'src/validate_xgboost.py',
]
all_passed = True
for s in scripts:
    res = subprocess.run([sys.executable, s])
    if res.returncode != 0:
        all_passed = False
assert all_passed, 'Validation failures detected!'
print('=' * 50)
print('ALL 6 PROJECT VALIDATION SUITES PASSED (0 ERRORS)')
print('=' * 50)
"
```







