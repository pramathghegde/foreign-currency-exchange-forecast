# Data Sources & Feasibility Specification

This document details all data series collected and cached for the Stanford CS229 (2019) research paper replication: *"Foreign Currency Exchange Rate Forecast"*.

Study Period: **January 1, 2001 to December 31, 2018** (Daily trading frequency).  
Data Directory: `data/raw/`  
Automated Loader: `src/data_loader.py`  
Validation Suite: `src/validate_data.py`

---

## 1. Currency Exchange Rates (Primary Source: FRED)

The original paper utilized Bloomberg Terminal daily close prices. For full public reproducibility without the 2001–2003 coverage gap present on Yahoo Finance, official daily exchange rates are obtained directly from the **Board of Governors of the Federal Reserve System (H.10 Release via FRED)**.

| Series ID | File Name | Quoted Currency Pair | Quote Convention / Units | Date Coverage | Total Rows | Unobserved (Holidays) | Valid Numeric Rows | Min Rate | Max Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DEXUSUK` | `fred_DEXUSUK.csv` | **GBP / USD** | USD per 1 GBP ($) | 2001-01-01 to 2018-12-31 | 4,696 | 178 | 4,518 | 1.2118 | 2.1104 |
| `DEXUSEU` | `fred_DEXUSEU.csv` | **EUR / USD** | USD per 1 EUR ($) | 2001-01-01 to 2018-12-31 | 4,696 | 178 | 4,518 | 0.8370 | 1.6010 |
| `DEXJPUS` | `fred_DEXJPUS.csv` | **USD / JPY** | JPY per 1 USD (¥) | 2001-01-01 to 2018-12-31 | 4,696 | 178 | 4,518 | 75.720 | 134.77 |

> [!IMPORTANT]
> **Exchange Rate Quote Conventions:**
> - `DEXUSUK` gives USD per GBP ($1.21 – $2.11). An increase means GBP strengthens and USD weakens.
> - `DEXUSEU` gives USD per EUR ($0.84 – $1.60). An increase means EUR strengthens and USD weakens. Note: In the paper's terminology "USD/EUR" is referenced, but the pricing unit is EUR/USD standard market quote.
> - `DEXJPUS` gives JPY per USD (¥75.72 – ¥134.77). An increase means USD strengthens and JPY weakens.
> 
> Missing observations (marked as `.` in raw FRED files) correspond to official Federal Reserve market holidays and have been preserved without artificial interpolation.

---

## 2. Financial Market Indices (Primary Source: Yahoo Finance)

The four equity market indices referenced in the Stanford paper were retrieved from Yahoo Finance via the cached loader:

| Ticker | File Name | Index Name | Columns Preserved | Date Coverage | Total Trading Days | Missing Values | Min Close | Max Close |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `^GSPC` | `yfinance_GSPC.csv` | **S&P 500** | Open, High, Low, Close, Adj Close, Volume | 2001-01-02 to 2018-12-31 | 4,527 | 0 | 676.53 | 2,930.75 |
| `^OEX` | `yfinance_OEX.csv` | **S&P 100** | Open, High, Low, Close, Adj Close, Volume | 2001-01-02 to 2018-12-31 | 4,527 | 0 | 322.13 | 1,302.80 |
| `^IXIC` | `yfinance_IXIC.csv` | **NASDAQ Composite** | Open, High, Low, Close, Adj Close, Volume | 2001-01-02 to 2018-12-31 | 4,527 | 0 | 1,114.11 | 8,109.69 |
| `^DJI` | `yfinance_DJI.csv` | **Dow Jones Industrial Average** | Open, High, Low, Close, Adj Close, Volume | 2001-01-02 to 2018-12-31 | 4,527 | 0 | 6,547.05 | 26,828.39 |

---

## 3. Macroeconomic Indicators (Primary Source: FRED)

The macroeconomic series specified in Section 3.1 of the paper are downloaded and cached from FRED:

| Series ID | File Name | Indicator Description | Native Frequency | Unit / Scale | Date Coverage | Observations | Min Value | Max Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DFF` | `fred_DFF.csv` | **Federal Funds Effective Rate** | Daily (7 days/wk) | Percent (%) | 2001-01-01 to 2018-12-31 | 6,574 | 0.04% | 6.67% |
| `CPIAUCNS` | `fred_CPIAUCNS.csv` | **Consumer Price Index (CPI)** | Monthly | Index 1982-84=100 | 2001-01-01 to 2018-12-01 | 216 | 175.10 | 252.885 |
| `USREC` | `fred_USREC.csv` | **NBER Recession Indicator** | Monthly | Binary (0 or 1) | 2001-01-01 to 2018-12-01 | 216 | 0.0 | 1.0 |
| `GFDEBTN` | `fred_GFDEBTN.csv` | **Federal Debt: Total Public Debt** | Quarterly | Millions of USD | 2001-01-01 to 2018-10-01 | 72 | \$5,726,815M | \$21,974,096M |

---

## 4. Time-Series Alignment & Prevention of Lookahead Bias

1. **Publication Lag Handling:**
   - `CPIAUCNS` is observed monthly and published mid-month following the observation period.
   - `GFDEBTN` is observed quarterly and published after the fiscal quarter closes.
   - In Step 4 (Feature Engineering), these series must only be forward-filled (`ffill()`) after their information release date onto active trading days. **Never apply backward-filling (`bfill()`) or rolling calculations that incorporate $t+1$ values.**
2. **Holiday Mismatches:**
   - `DEXUSUK`, `DEXUSEU`, and `DEXJPUS` each have 178 unobserved holiday days across the 18-year history.
   - Per paper Section 5.2: Days where all markets are closed are skipped. Days where a specific regional market is closed are handled by trading exclusively among open currency pairs.
3. **Reproducibility & Caching:**
   - All raw datasets are cached in `data/raw/`. Re-executing `python src/data_loader.py` validates existing caches without redownloading unless called with `--force`.
