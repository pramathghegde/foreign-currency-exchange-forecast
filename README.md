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

## 6. Repository Structure

```text
├── data/
│   ├── raw/                 # Downloaded raw historical datasets (11 series + metadata.json)
│   ├── processed/           # Aligned features and cleaned series (.gitkeep)
│   └── DATA_SOURCES.md      # Detailed source documentation and ticker mapping
├── notebooks/               # Interactive exploration and prototyping notebooks (.gitkeep)
├── reports/                 # Project documentation and milestone reports (.gitkeep)
├── results/
│   ├── figures/             # Exported performance curves and cash plots (.gitkeep)
│   └── tables/              # Exported comparative metric tables (CSV/LaTeX) (.gitkeep)
├── src/
│   ├── __init__.py          # Source package initializer
│   ├── data_loader.py       # Automated data downloader with local file caching
│   └── validate_data.py     # Data validation and integrity checking suite
├── .gitignore               # Excludes caches, venvs, and generated outputs
├── requirements.txt         # Minimal project dependencies
└── README.md                # Project documentation and reproduction guide
```

---

## 7. Next Steps: Feature Engineering & Preprocessing

Once raw data collection and validation are complete, the upcoming milestone (Step 4) will implement `src/feature_engineering.py` to:
1. Construct the 15 features specified in the Stanford paper (log returns, moving averages of returns, polynomial terms, moving averages of prices).
2. Align macroeconomic variables forward without lookahead bias.
3. Generate chronological 80% train, 10% validation, and 10% test splits.
