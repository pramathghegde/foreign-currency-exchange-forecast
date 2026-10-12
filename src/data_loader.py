"""
src/data_loader.py

Modular historical data downloader and caching utility for the Stanford CS229 (2019)
Foreign Currency Exchange Rate Forecast replication and extension.

Downloads and caches raw daily data from 2001-01-01 to 2018-12-31:
- FX Rates (FRED): DEXUSUK (GBP/USD), DEXUSEU (EUR/USD), DEXJPUS (USD/JPY)
- Financial Indices (Yahoo Finance): ^GSPC, ^OEX, ^IXIC, ^DJI
- Macroeconomic Series (FRED): DFF, CPIAUCNS, USREC, GFDEBTN
"""

import os
import io
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from curl_cffi import requests as crequests
except ImportError:
    crequests = None

import requests
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
START_DATE = "2001-01-01"
END_DATE = "2018-12-31"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

# Series Definitions
FRED_SERIES = {
    # Currency Exchange Rates (Primary official daily Board of Governors series)
    "DEXUSUK": {
        "name": "GBP/USD Exchange Rate",
        "quote_convention": "USD per 1 GBP (U.S. Dollars per British Pound)",
        "frequency": "Daily",
        "category": "fx_rate",
    },
    "DEXUSEU": {
        "name": "EUR/USD Exchange Rate",
        "quote_convention": "USD per 1 EUR (U.S. Dollars per Euro)",
        "frequency": "Daily",
        "category": "fx_rate",
    },
    "DEXJPUS": {
        "name": "USD/JPY Exchange Rate",
        "quote_convention": "JPY per 1 USD (Japanese Yen per U.S. Dollar)",
        "frequency": "Daily",
        "category": "fx_rate",
    },
    # Macroeconomic Series
    "DFF": {
        "name": "Federal Funds Effective Rate",
        "quote_convention": "Percent",
        "frequency": "Daily",
        "category": "macro",
    },
    "CPIAUCNS": {
        "name": "Consumer Price Index: All Urban Consumers",
        "quote_convention": "Index 1982-1984=100 (Unadjusted)",
        "frequency": "Monthly",
        "category": "macro",
    },
    "USREC": {
        "name": "NBER Recession Indicator",
        "quote_convention": "Binary indicator (1 = Recession, 0 = Expansion)",
        "frequency": "Monthly",
        "category": "macro",
    },
    "GFDEBTN": {
        "name": "Federal Debt: Total Public Debt",
        "quote_convention": "Millions of Dollars (Quarterly)",
        "frequency": "Quarterly",
        "category": "macro",
    },
}

YFINANCE_SERIES = {
    "^GSPC": {
        "name": "S&P 500 Index",
        "filename": "yfinance_GSPC.csv",
        "category": "market_index",
    },
    "^OEX": {
        "name": "S&P 100 Index",
        "filename": "yfinance_OEX.csv",
        "category": "market_index",
    },
    "^IXIC": {
        "name": "Nasdaq Composite",
        "filename": "yfinance_IXIC.csv",
        "category": "market_index",
    },
    "^DJI": {
        "name": "Dow Jones Industrial Average",
        "filename": "yfinance_DJI.csv",
        "category": "market_index",
    },
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}


def download_fred_series(
    series_id: str,
    raw_dir: str = RAW_DATA_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    force_refresh: bool = False,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Download a single FRED series via public CSV endpoint and cache locally.
    Uses curl_cffi (with Chrome TLS fingerprint) to avoid Cloudflare read timeouts.
    Preserves raw data, dates, and missing observation markers (e.g., '.').
    """
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, f"fred_{series_id}.csv")
    meta = FRED_SERIES.get(series_id, {})

    if os.path.exists(out_path) and not force_refresh:
        logger.info(f"[CACHED] FRED {series_id} already exists at {out_path}")
        df = pd.read_csv(out_path)
        return _summarize_fred_dataframe(df, series_id, out_path, meta, cached=True)

    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    logger.info(f"[DOWNLOADING] FRED {series_id} from {url}...")

    csv_text = None
    error_msg = None

    # Priority 1: curl_cffi with Chrome impersonation (bypasses Cloudflare bot challenges)
    if crequests is not None:
        try:
            resp = crequests.get(url, impersonate="chrome120", timeout=timeout)
            if resp.status_code == 200:
                csv_text = resp.text
            else:
                error_msg = f"HTTP {resp.status_code}"
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"curl_cffi request failed for {series_id}: {e}")

    # Priority 2: Standard requests fallback
    if csv_text is None:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            csv_text = resp.text
        except Exception as e:
            logger.error(f"[FAILED] FRED {series_id} download error: {e}")
            return {
                "series_id": series_id,
                "source": "FRED",
                "file_path": out_path,
                "status": "FAILED",
                "error": error_msg or str(e),
            }

    # Parse CSV content
    try:
        df_raw = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        logger.error(f"[FAILED] Failed to parse CSV for {series_id}: {e}")
        return {
            "series_id": series_id,
            "source": "FRED",
            "file_path": out_path,
            "status": "FAILED",
            "error": str(e),
        }

    if len(df_raw.columns) < 2:
        err = f"Unexpected format from FRED for {series_id}: columns={list(df_raw.columns)}"
        logger.error(err)
        return {
            "series_id": series_id,
            "source": "FRED",
            "file_path": out_path,
            "status": "FAILED",
            "error": err,
        }

    date_col = df_raw.columns[0]
    val_col = df_raw.columns[1]

    # Standardize column naming: DATE, <SERIES_ID>
    df_raw.rename(columns={date_col: "DATE", val_col: series_id}, inplace=True)
    df_raw["DATE"] = pd.to_datetime(df_raw["DATE"], errors="coerce")
    df_raw = df_raw.dropna(subset=["DATE"])

    # Filter target window [start_date, end_date]
    mask = (df_raw["DATE"] >= start_date) & (df_raw["DATE"] <= end_date)
    df_filtered = df_raw.loc[mask].copy()
    df_filtered["DATE"] = df_filtered["DATE"].dt.strftime("%Y-%m-%d")

    # Save raw CSV
    df_filtered.to_csv(out_path, index=False)
    logger.info(f"[SAVED] FRED {series_id} ({len(df_filtered)} rows) to {out_path}")

    return _summarize_fred_dataframe(df_filtered, series_id, out_path, meta, cached=False)


def _summarize_fred_dataframe(
    df: pd.DataFrame, series_id: str, out_path: str, meta: Dict[str, Any], cached: bool
) -> Dict[str, Any]:
    """Helper to compute validation summary on FRED series."""
    date_series = pd.to_datetime(df["DATE"])
    val_series = df[series_id]

    # In FRED data, '.' denotes non-trading day / missing value
    missing_dots = (val_series == ".").sum()
    missing_nans = val_series.isna().sum()
    total_missing = int(missing_dots + missing_nans)

    numeric_vals = pd.to_numeric(val_series.replace(".", None), errors="coerce").dropna()
    min_val = float(numeric_vals.min()) if not numeric_vals.empty else None
    max_val = float(numeric_vals.max()) if not numeric_vals.empty else None

    return {
        "series_id": series_id,
        "name": meta.get("name", series_id),
        "source": "FRED",
        "file_path": out_path,
        "status": "SUCCESS",
        "cached": cached,
        "category": meta.get("category", "unknown"),
        "quote_convention": meta.get("quote_convention", ""),
        "native_frequency": meta.get("frequency", ""),
        "row_count": len(df),
        "start_date": str(date_series.min().strftime("%Y-%m-%d")) if not df.empty else None,
        "end_date": str(date_series.max().strftime("%Y-%m-%d")) if not df.empty else None,
        "missing_count": total_missing,
        "valid_count": int(len(numeric_vals)),
        "min_value": min_val,
        "max_value": max_val,
    }


def download_yfinance_series(
    ticker: str,
    raw_dir: str = RAW_DATA_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    force_refresh: bool = False,
    timeout: int = 15,
) -> Dict[str, Any]:
    """
    Download a single financial index from Yahoo Finance and cache locally.
    Preserves Open, High, Low, Close, Adj Close, Volume.
    """
    os.makedirs(raw_dir, exist_ok=True)
    meta = YFINANCE_SERIES.get(ticker, {})
    filename = meta.get("filename", f"yfinance_{ticker.replace('^', '')}.csv")
    out_path = os.path.join(raw_dir, filename)

    if os.path.exists(out_path) and not force_refresh:
        logger.info(f"[CACHED] Yahoo Finance {ticker} already exists at {out_path}")
        df = pd.read_csv(out_path)
        return _summarize_yf_dataframe(df, ticker, out_path, meta, cached=True)

    logger.info(f"[DOWNLOADING] Yahoo Finance {ticker} ({start_date} to {end_date})...")

    # End date in yfinance is exclusive, so add 1 day to include END_DATE
    end_date_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
    end_date_str = end_date_dt.strftime("%Y-%m-%d")

    df = None
    error_msg = None

    if yf is not None:
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date_str,
                progress=False,
                auto_adjust=False,
                timeout=timeout,
            )
            # Handle possible MultiIndex columns from recent yfinance versions
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"yfinance download failed for {ticker}: {e}. Trying direct chart API fallback...")

    # Fallback to direct Yahoo Finance chart API if yfinance fails
    if df is None or df.empty:
        try:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) + 86400
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"
            res = requests.get(url, headers=HEADERS, timeout=timeout)
            res.raise_for_status()
            data = res.json()["chart"]["result"][0]
            timestamps = data.get("timestamp", [])
            indicators = data["indicators"]["quote"][0]
            adj_close = data["indicators"].get("adjclose", [{}])[0].get("adjclose", indicators["close"])

            dates = [datetime.fromtimestamp(ts).strftime("%Y-%m-%d") for ts in timestamps]
            df = pd.DataFrame(
                {
                    "Open": indicators.get("open", []),
                    "High": indicators.get("high", []),
                    "Low": indicators.get("low", []),
                    "Close": indicators.get("close", []),
                    "Adj Close": adj_close,
                    "Volume": indicators.get("volume", []),
                },
                index=pd.to_datetime(dates),
            )
            df.index.name = "Date"
        except Exception as e:
            logger.error(f"[FAILED] Direct chart API fallback failed for {ticker}: {e}")
            return {
                "series_id": ticker,
                "source": "Yahoo Finance",
                "file_path": out_path,
                "status": "FAILED",
                "error": error_msg or str(e),
            }

    if df is None or df.empty:
        return {
            "series_id": ticker,
            "source": "Yahoo Finance",
            "file_path": out_path,
            "status": "FAILED",
            "error": "No data returned for ticker",
        }

    df = df.reset_index()
    # Standardize Date column
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df.rename(columns={date_col: "DATE"}, inplace=True)
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m-%d")

    # Filter to requested range
    mask = (df["DATE"] >= start_date) & (df["DATE"] <= end_date)
    df_filtered = df.loc[mask].copy()

    df_filtered.to_csv(out_path, index=False)
    logger.info(f"[SAVED] Yahoo Finance {ticker} ({len(df_filtered)} rows) to {out_path}")

    return _summarize_yf_dataframe(df_filtered, ticker, out_path, meta, cached=False)


def _summarize_yf_dataframe(
    df: pd.DataFrame, ticker: str, out_path: str, meta: Dict[str, Any], cached: bool
) -> Dict[str, Any]:
    """Helper to compute summary on Yahoo Finance DataFrame."""
    date_series = pd.to_datetime(df["DATE"])
    close_series = df["Close"] if "Close" in df.columns else df.iloc[:, 1]
    missing_count = int(close_series.isna().sum())
    valid_count = int(close_series.dropna().count())
    min_val = float(close_series.min()) if valid_count > 0 else None
    max_val = float(close_series.max()) if valid_count > 0 else None

    return {
        "series_id": ticker,
        "name": meta.get("name", ticker),
        "source": "Yahoo Finance",
        "file_path": out_path,
        "status": "SUCCESS",
        "cached": cached,
        "category": meta.get("category", "market_index"),
        "quote_convention": "Index Points",
        "native_frequency": "Daily",
        "row_count": len(df),
        "start_date": str(date_series.min().strftime("%Y-%m-%d")) if not df.empty else None,
        "end_date": str(date_series.max().strftime("%Y-%m-%d")) if not df.empty else None,
        "missing_count": missing_count,
        "valid_count": valid_count,
        "min_value": min_val,
        "max_value": max_val,
    }


def download_all_series(
    raw_dir: str = RAW_DATA_DIR,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Downloads and caches all 11 required project series.
    Saves a comprehensive metadata summary to data/raw/metadata.json.
    """
    os.makedirs(raw_dir, exist_ok=True)
    summary = {}

    logger.info("=" * 70)
    logger.info("STARTING DATA INGESTION: 2001-01-01 to 2018-12-31")
    logger.info("=" * 70)

    # 1. FRED Exchange Rates
    logger.info("\n--- 1. CURRENCY EXCHANGE RATES (FRED) ---")
    for sid in ["DEXUSUK", "DEXUSEU", "DEXJPUS"]:
        summary[sid] = download_fred_series(
            series_id=sid,
            raw_dir=raw_dir,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh,
        )

    # 2. Yahoo Finance Financial Indices
    logger.info("\n--- 2. FINANCIAL MARKET INDICES (Yahoo Finance) ---")
    for ticker in ["^GSPC", "^OEX", "^IXIC", "^DJI"]:
        summary[ticker] = download_yfinance_series(
            ticker=ticker,
            raw_dir=raw_dir,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh,
        )

    # 3. FRED Macroeconomic Series
    logger.info("\n--- 3. MACROECONOMIC FACTORS (FRED) ---")
    for sid in ["DFF", "CPIAUCNS", "USREC", "GFDEBTN"]:
        summary[sid] = download_fred_series(
            series_id=sid,
            raw_dir=raw_dir,
            start_date=start_date,
            end_date=end_date,
            force_refresh=force_refresh,
        )

    # Save summary metadata
    meta_path = os.path.join(raw_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nMetadata summary recorded at: {meta_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and cache historical data.")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached.")
    args = parser.parse_args()

    results = download_all_series(force_refresh=args.force)
    print("\nSummary Results:")
    for k, v in results.items():
        print(f"  {k:<10}: {v.get('status')} | Rows={v.get('row_count')} | Range={v.get('start_date')} to {v.get('end_date')}")
