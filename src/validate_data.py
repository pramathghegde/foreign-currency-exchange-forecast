"""
src/validate_data.py

Validation script to inspect downloaded raw data files in data/raw/
for the Stanford CS229 (2019) Foreign Currency Exchange Rate Forecast replication.

Checks:
- Existence of each expected raw file.
- Date coverage and row counts (target window: 2001-01-01 to 2018-12-31).
- Detection of duplicate dates.
- Missing values (NaNs, empty strings, FRED '.' markers).
- Validity of exchange rates (must be strictly positive numeric values).
- Validity of financial indices and macro indicators.
"""

import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

EXPECTED_FILES = {
    # FX Rates (FRED)
    "fred_DEXUSUK.csv": {
        "series_id": "DEXUSUK",
        "name": "GBP/USD Exchange Rate",
        "category": "fx_rate",
        "expected_freq": "Daily",
        "val_col": "DEXUSUK",
        "is_fx": True,
        "min_expected_rows": 4500,
    },
    "fred_DEXUSEU.csv": {
        "series_id": "DEXUSEU",
        "name": "EUR/USD Exchange Rate",
        "category": "fx_rate",
        "expected_freq": "Daily",
        "val_col": "DEXUSEU",
        "is_fx": True,
        "min_expected_rows": 4500,
    },
    "fred_DEXJPUS.csv": {
        "series_id": "DEXJPUS",
        "name": "USD/JPY Exchange Rate",
        "category": "fx_rate",
        "expected_freq": "Daily",
        "val_col": "DEXJPUS",
        "is_fx": True,
        "min_expected_rows": 4500,
    },
    # Financial Indices (Yahoo Finance)
    "yfinance_GSPC.csv": {
        "series_id": "^GSPC",
        "name": "S&P 500 Index",
        "category": "market_index",
        "expected_freq": "Daily",
        "val_col": "Close",
        "is_fx": False,
        "min_expected_rows": 4400,
    },
    "yfinance_OEX.csv": {
        "series_id": "^OEX",
        "name": "S&P 100 Index",
        "category": "market_index",
        "expected_freq": "Daily",
        "val_col": "Close",
        "is_fx": False,
        "min_expected_rows": 4400,
    },
    "yfinance_IXIC.csv": {
        "series_id": "^IXIC",
        "name": "Nasdaq Composite",
        "category": "market_index",
        "expected_freq": "Daily",
        "val_col": "Close",
        "is_fx": False,
        "min_expected_rows": 4400,
    },
    "yfinance_DJI.csv": {
        "series_id": "^DJI",
        "name": "Dow Jones Industrial Average",
        "category": "market_index",
        "expected_freq": "Daily",
        "val_col": "Close",
        "is_fx": False,
        "min_expected_rows": 4400,
    },
    # Macroeconomic Series (FRED)
    "fred_DFF.csv": {
        "series_id": "DFF",
        "name": "Federal Funds Effective Rate",
        "category": "macro",
        "expected_freq": "Daily",
        "val_col": "DFF",
        "is_fx": False,
        "min_expected_rows": 6500,
    },
    "fred_CPIAUCNS.csv": {
        "series_id": "CPIAUCNS",
        "name": "Consumer Price Index: All Urban Consumers",
        "category": "macro",
        "expected_freq": "Monthly",
        "val_col": "CPIAUCNS",
        "is_fx": False,
        "min_expected_rows": 210,
    },
    "fred_USREC.csv": {
        "series_id": "USREC",
        "name": "NBER Recession Indicator",
        "category": "macro",
        "expected_freq": "Monthly",
        "val_col": "USREC",
        "is_fx": False,
        "min_expected_rows": 210,
    },
    "fred_GFDEBTN.csv": {
        "series_id": "GFDEBTN",
        "name": "Federal Debt: Total Public Debt",
        "category": "macro",
        "expected_freq": "Quarterly",
        "val_col": "GFDEBTN",
        "is_fx": False,
        "min_expected_rows": 70,
    },
}


def validate_file(filename: str, spec: Dict[str, Any], raw_dir: str = RAW_DATA_DIR) -> Dict[str, Any]:
    """Inspect and validate a single raw dataset."""
    file_path = os.path.join(raw_dir, filename)

    if not os.path.exists(file_path):
        return {
            "file": filename,
            "series_id": spec["series_id"],
            "name": spec["name"],
            "exists": False,
            "status": "MISSING_FILE",
            "errors": [f"File {filename} does not exist at {file_path}"],
        }

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {
            "file": filename,
            "series_id": spec["series_id"],
            "name": spec["name"],
            "exists": True,
            "status": "UNREADABLE",
            "errors": [f"Failed to read CSV: {e}"],
        }

    errors: List[str] = []
    warnings: List[str] = []

    # 1. Date column checks
    if "DATE" not in df.columns:
        errors.append(f"Missing required 'DATE' column. Found columns: {list(df.columns)}")
        date_series = pd.Series([], dtype="object")
    else:
        date_series = pd.to_datetime(df["DATE"], errors="coerce")
        invalid_dates = date_series.isna().sum()
        if invalid_dates > 0:
            errors.append(f"Found {invalid_dates} unparseable dates in DATE column.")

        # Duplicate dates check
        duplicates = df["DATE"].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate dates.")

    # 2. Row count check
    row_count = len(df)
    if row_count < spec["min_expected_rows"]:
        warnings.append(
            f"Row count ({row_count}) is below expected minimum ({spec['min_expected_rows']})."
        )

    # 3. Target value column checks
    val_col = spec["val_col"]
    if val_col not in df.columns:
        errors.append(f"Expected value column '{val_col}' not found. Available: {list(df.columns)}")
        numeric_series = pd.Series([], dtype="float64")
        missing_count = 0
        non_positive_count = 0
    else:
        raw_vals = df[val_col]

        # In FRED data, '.' represents missing/holiday observations
        missing_dots = (raw_vals == ".").sum() if raw_vals.dtype == object else 0
        missing_nans = raw_vals.isna().sum()
        missing_count = int(missing_dots + missing_nans)

        cleaned_vals = raw_vals.replace(".", None) if raw_vals.dtype == object else raw_vals
        numeric_series = pd.to_numeric(cleaned_vals, errors="coerce")

        # 4. Validity & Positive Rate checks
        non_positive_count = int((numeric_series <= 0).sum())
        if spec["is_fx"]:
            if non_positive_count > 0:
                errors.append(f"Found {non_positive_count} non-positive exchange rate values (<= 0).")
        elif spec["category"] == "market_index":
            if non_positive_count > 0:
                errors.append(f"Found {non_positive_count} non-positive index values (<= 0).")

    start_date = str(date_series.min().strftime("%Y-%m-%d")) if not date_series.empty else "N/A"
    end_date = str(date_series.max().strftime("%Y-%m-%d")) if not date_series.empty else "N/A"

    min_val = float(numeric_series.min()) if not numeric_series.dropna().empty else None
    max_val = float(numeric_series.max()) if not numeric_series.dropna().empty else None

    status = "PASSED" if not errors else "FAILED"
    if status == "PASSED" and warnings:
        status = "PASSED_WITH_WARNINGS"

    return {
        "file": filename,
        "series_id": spec["series_id"],
        "name": spec["name"],
        "category": spec["category"],
        "exists": True,
        "status": status,
        "row_count": row_count,
        "start_date": start_date,
        "end_date": end_date,
        "duplicate_dates": int(df["DATE"].duplicated().sum()) if "DATE" in df.columns else 0,
        "missing_values": missing_count,
        "valid_numeric_count": int(numeric_series.dropna().count()),
        "non_positive_count": non_positive_count,
        "min_value": min_val,
        "max_value": max_val,
        "warnings": warnings,
        "errors": errors,
    }


def validate_all(raw_dir: str = RAW_DATA_DIR) -> Dict[str, Any]:
    """Validate all expected raw files in data/raw/."""
    results = {}
    total_passed = 0
    total_failed = 0

    print("=" * 80)
    print("VALIDATING RAW DATA REPOSITORY (data/raw/)")
    print(f"Target Directory: {raw_dir}")
    print("=" * 80)

    for filename, spec in EXPECTED_FILES.items():
        res = validate_file(filename, spec, raw_dir=raw_dir)
        results[filename] = res
        if res["status"] in ["PASSED", "PASSED_WITH_WARNINGS"]:
            total_passed += 1
            status_symbol = "[OK]"
        else:
            total_failed += 1
            status_symbol = "[FAIL]"

        print(f"\n{status_symbol} {filename} ({spec['name']}):")
        print(f"     Status:           {res['status']}")
        print(f"     Row Count:        {res.get('row_count', 'N/A')}")
        print(f"     Date Coverage:    {res.get('start_date', 'N/A')} to {res.get('end_date', 'N/A')}")
        print(f"     Duplicate Dates:  {res.get('duplicate_dates', 'N/A')}")
        print(f"     Missing Values:   {res.get('missing_values', 'N/A')} (unobserved/holidays)")
        print(f"     Numeric Range:    min={res.get('min_value')}, max={res.get('max_value')}")

        if res.get("warnings"):
            for w in res["warnings"]:
                print(f"     WARNING: {w}")
        if res.get("errors"):
            for e in res["errors"]:
                print(f"     ERROR:   {e}")

    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {total_passed} / {len(EXPECTED_FILES)} files PASSED.")
    if total_failed > 0:
        print(f"CRITICAL: {total_failed} files FAILED or are missing.")
    else:
        print("ALL RAW DATA FILES VERIFIED AND READY FOR DOWNSTREAM PROCESSING.")
    print("=" * 80)

    return {
        "total_files": len(EXPECTED_FILES),
        "passed": total_passed,
        "failed": total_failed,
        "details": results,
    }


if __name__ == "__main__":
    import sys

    summary = validate_all()
    if summary["failed"] > 0:
        sys.exit(1)
    sys.exit(0)
