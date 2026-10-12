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

1. **Macroeconomic Publication Lag Modeling:**
   - **Consumer Price Index (`CPIAUCNS`):** The Bureau of Labor Statistics (BLS) publishes monthly CPI data mid-month following the observation period. In our pipeline, availability date is calculated strictly as:
     $$\text{Available Date} = \text{Observation Month End} + 15\text{ calendar days}$$
     For example, January 2001 CPI (observed 2001-01-01, ending 2001-01-31) is available to the model starting 2001-02-15.
   - **NBER Recession Indicator (`USREC`):** Observed monthly on the 1st of the month. Availability date is calculated as:
     $$\text{Available Date} = \text{Observation Month End} + 30\text{ calendar days}$$
     *Assumption Note:* In economic history, official NBER Business Cycle Dating Committee declarations can take 6–18 months. Using `USREC` with a 30-day lag follows standard econometric backtesting practice as used in the reference paper, but represents an ex-post availability approximation.
   - **Federal Debt (`GFDEBTN`):** Total public debt is observed quarterly on the first day of each fiscal quarter. Availability date is calculated strictly from the **end of the observation quarter**:
     $$\text{Available Date} = \text{Observation Quarter End} + 45\text{ calendar days}$$
     For example, Q1 2001 debt (observed 2001-01-01, quarter ending 2001-03-31) is available to the model starting 2001-05-15.
   - All macroeconomic series are merged onto the daily trading calendar strictly using forward-fill (`ffill()`) without backward filling or rolling lookahead.

2. **Currency Quote Directions & Target Consistency:**
   - **GBP/USD (`DEXUSUK`):** Quoted as **USD per 1 GBP**. Higher rate = GBP appreciates, USD depreciates.
   - **EUR/USD (`DEXUSEU`):** Quoted as **USD per 1 EUR**. Higher rate = EUR appreciates, USD depreciates.
   - **USD/JPY (`DEXJPUS`):** Quoted as **JPY per 1 USD**. Higher rate = USD appreciates, JPY depreciates.
   - **Target Definitions:**
     - `target_reg` = $\ln(P_{t+1} / P_t)$ (next-day log return of quoted price).
     - `target_clf` = $1$ if $P_{t+1} > P_t$ else $0$ (next-day upward movement of quoted price).
     - `target_ratio` = $\frac{P_{t+1}}{P_t} = \exp(\text{target\_reg})$.
   - **Trading Simulation Interpretation:**
     In the paper's trading simulation, the investor holds USD and switches into foreign currency when USD is projected to depreciate against that currency:
     - For JPY (quoted in USD terms): USD depreciates when $\frac{P_{t+1}}{P_t} < 1 \iff r_{t+1} < 0$.
     - For GBP & EUR (quoted in foreign currency terms): USD depreciates when $\frac{P_t}{P_{t+1}} < 1 \iff r_{t+1} > 0$.

3. **Holiday Mismatches:**
   - `DEXUSUK`, `DEXUSEU`, and `DEXJPUS` each have 178 unobserved Federal Reserve bank holidays across 2001–2018.
   - These dates are retained without artificial linear interpolation. Rolling windows skip non-trading dates and track changes across active trading sessions.

4. **Reproducibility & Determinism:**
   - All raw files are cached in `data/raw/`.
   - Feature engineering produces identical MD5 checksums across all 21 processed CSV datasets upon re-execution.

