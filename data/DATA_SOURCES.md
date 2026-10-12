# Data Sources & Feasibility Specification

This document details all data series required to reproduce and extend the Stanford CS229 (2019) research paper *"Foreign Currency Exchange Rate Forecast"*.

Expected study period: **January 1, 2001 to December 31, 2018** (Daily trading frequency).

---

## 1. Financial Market Indices

The original paper utilized U.S. financial equity market indices sourced from CRSP (Center for Research in Security Prices). These series are publicly verifiable and available at daily resolution via Yahoo Finance (`yfinance` / chart API).

| Asset Name | Primary Ticker (Yahoo Finance) | Frequency | Date Range Verified (2001–2018) | Observations Returned | Status & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S&P 500** | `^GSPC` | Daily | 2001-01-02 to 2018-12-28 | 4,526 trading days | **Complete**. Covers entire study window. Standard NYSE/NASDAQ holiday calendar. |
| **S&P 100** | `^OEX` | Daily | 2001-01-02 to 2018-12-28 | 4,526 trading days | **Complete**. Covers entire study window. |
| **NASDAQ Composite** | `^IXIC` | Daily | 2001-01-02 to 2018-12-28 | 4,526 trading days | **Complete**. Covers entire study window. |
| **Dow Jones Industrial Average** | `^DJI` | Daily | 2001-01-02 to 2018-12-28 | 4,526 trading days | **Complete**. Covers entire study window. |

---

## 2. Foreign Exchange Rates

The original paper sourced daily foreign exchange closing prices from Bloomberg Terminal.

### Yahoo Finance Availability Check

| Currency Pair | Paper Expression | Yahoo Finance Ticker | Verified Date Range (Yahoo Finance) | Observations Returned | Status & Findings |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GBP / USD** | GBP/USD | `GBPUSD=X` | 2003-12-01 to 2018-12-28 | 3,935 daily quotes | **Partial**. Yahoo Finance historical records for GBP/USD begin on Dec 1, 2003. Observations from 2001-01-01 to 2003-11-30 are missing on Yahoo Finance. |
| **USD / EUR** | USD/EUR | `EURUSD=X` | 2003-12-01 to 2018-12-28 | 3,935 daily quotes | **Partial**. Yahoo Finance historical records for EUR/USD begin on Dec 1, 2003. (Note: Yahoo quotes EUR/USD as USD per EUR; inverted quote gives USD/EUR). |
| **USD / JPY** | USD/JPY | `JPY=X` | 2001-01-01 to 2018-12-28 | 4,695 daily quotes | **Complete**. Full 2001–2018 daily history available on Yahoo Finance. |

### Public Substitute / Backfill for 2001–2003 FX Data: Federal Reserve (FRED)

To overcome the 2001–2003 gap on Yahoo Finance without fabricating data, the Board of Governors of the Federal Reserve System (H.10 Foreign Exchange Rates release via FRED) provides verified daily official rates:

| Currency Pair | FRED Series ID | Verified Date Range (FRED) | Observations (2001–2018) | Role / Solution |
| :--- | :--- | :--- | :--- | :--- |
| **U.S. / U.K. Foreign Exchange Rate** | `DEXUSUK` | 2001-01-01 to 2018-12-31 | 4,696 daily records | Supplies daily GBP/USD rates for the full 2001–2018 window (including the 2001–2003 gap). |
| **U.S. / Euro Foreign Exchange Rate** | `DEXUSEU` | 2001-01-01 to 2018-12-31 | 4,696 daily records | Supplies daily EUR/USD rates for the full 2001–2018 window (including the 2001–2003 gap). |
| **Japan / U.S. Foreign Exchange Rate** | `DEXJPUS` | 2001-01-01 to 2018-12-31 | 4,696 daily records | Corroborates daily USD/JPY rates for the full 2001–2018 window. |

---

## 3. Macroeconomic Indicators

The original paper utilized four core macroeconomic factors from CRSP / Federal Reserve records. These are verified and directly downloadable via the Federal Reserve Bank of St. Louis (FRED):

| Indicator Name | FRED Series ID | Native Frequency | Verified Date Range (FRED) | Observations (2001–2018) | Processing & Alignment Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer Price Index (CPI)** | `CPIAUCNS` *(Consumer Price Index for All Urban Consumers: All Items)* | Monthly | 2001-01-01 to 2018-12-01 | 216 monthly values | Calculate rate of change: $\Delta \text{CPI}_t = \frac{\text{CPI}_t - \text{CPI}_{t-1}}{\text{CPI}_{t-1}}$. Forward-fill (`ffill()`) onto daily trading dates. |
| **Federal Funds Rate** | `DFF` *(Federal Funds Effective Rate)* | Daily | 2001-01-01 to 2018-12-31 | 6,574 daily values | **Direct Daily Match**. Alternative monthly benchmark is `FEDFUNDS` (216 observations). |
| **NBER Recession Indicator** | `USREC` *(NBER based Recession Indicator for US)* | Monthly | 2001-01-01 to 2018-12-01 | 216 monthly values | Binary indicator (0 = expansion, 1 = recession). Forward-fill onto daily trading calendar. Daily series `USRECD` also exists. |
| **Federal Debt** | `GFDEBTN` *(Federal Debt: Total Public Debt)* | Quarterly | 2001-01-01 to 2018-10-01 | 72 quarterly values | Expressed in Millions of Dollars. Forward-fill (`ffill()`) to daily frequency based on release dates to avoid lookahead bias. |

---

## 4. Preservation of Chronology & Prevention of Lookahead Bias

1. **Strict Forward Alignment:**
   - Lower-frequency macroeconomic variables (monthly CPI, quarterly Federal Debt) must only be merged onto daily trading timestamps using historical release dates or trailing forward-fill (`ffill()`).
   - No backward filling (`bfill()`) or rolling calculations that incorporate future timestamps ($t+1$) into features at time $t$.
2. **Missing Trading Day Alignment:**
   - Bank holidays differ between London, Frankfurt, Tokyo, and New York.
   - Per Section 5.2 of the Stanford CS229 paper: if an exchange rate is not available on a particular day due to regional market closures, trade simulation only allocates across active markets for that day; if all markets are closed, the day is skipped.
3. **Partitioning:**
   - **Train:** 2001 to early 2017 (~80% of data).
   - **Validation:** 2017 (~10% of data).
   - **Test:** March 2017 through November 30, 2018 (~10% of data).
   - Model parameters, scalers (`StandardScaler`), and feature selectors must be fitted strictly on training data and applied out-of-sample to validation/test partitions.
