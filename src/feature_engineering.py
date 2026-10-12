"""
src/feature_engineering.py

Feature engineering and dataset preprocessing pipeline for the Stanford CS229 (2019)
Foreign Currency Exchange Rate Forecast replication and extension.

Constructs:
- 15 core technical features per Section 3.2 of the Stanford paper:
  * return_t: Current day log return
  * return_t_1: Previous day log return
  * return_ma_{5, 10, 20, 60, 120}: Moving averages of log returns
  * log_price: Natural logarithm of closing rate
  * log_price_sq: Squared log price
  * log_price_cube: Cubed log price
  * log_price_ma_{5, 10, 20, 60, 220}: Moving averages of log prices
- Next-day prediction targets:
  * target_reg: r_{t+1} = ln(P_{t+1} / P_t) (numerical log return)
  * target_clf: 1 if P_{t+1} > P_t else 0 (binary direction)
  * target_ratio: P_{t+1} / P_t = exp(r_{t+1}) (used in profit simulation)
- Financial & macroeconomic features with publication-lag alignment:
  * Daily equity index log returns (S&P 500, S&P 100, NASDAQ, DJIA)
  * Daily Federal Funds Effective Rate (DFF)
  * Monthly CPI rate of change with 15-day publication lag
  * Monthly NBER recession indicator with 30-day reporting lag
  * Quarterly public debt growth rate with 45-day reporting lag
- Chronological train/validation/test partitions matching paper test window:
  * EUR Test: 2017-03-03 to 2018-11-30 (440 days)
  * GBP/JPY Test: 2017-03-07 to 2018-11-30 (438 days)
  * Strict chronological split (~80% Train, ~10% Val, ~10% Test)
- Feature scaling (StandardScaler) fitted strictly on training partition only.
"""

import os
import json
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Currency Specifications
CURRENCY_CONFIG = {
    "GBP": {
        "raw_file": "fred_DEXUSUK.csv",
        "val_col": "DEXUSUK",
        "name": "GBP/USD",
        "quote_convention": "USD per 1 GBP (U.S. Dollars per British Pound)",
        "test_start": "2017-03-07",
        "test_end": "2018-11-30",
    },
    "EUR": {
        "raw_file": "fred_DEXUSEU.csv",
        "val_col": "DEXUSEU",
        "name": "EUR/USD",
        "quote_convention": "USD per 1 EUR (U.S. Dollars per Euro)",
        "test_start": "2017-03-03",
        "test_end": "2018-11-30",
    },
    "JPY": {
        "raw_file": "fred_DEXJPUS.csv",
        "val_col": "DEXJPUS",
        "name": "USD/JPY",
        "quote_convention": "JPY per 1 USD (Japanese Yen per U.S. Dollar)",
        "test_start": "2017-03-07",
        "test_end": "2018-11-30",
    },
}

TECHNICAL_FEATURE_COLS = [
    "return_t",
    "return_t_1",
    "return_ma_5",
    "return_ma_10",
    "return_ma_20",
    "return_ma_60",
    "return_ma_120",
    "log_price",
    "log_price_sq",
    "log_price_cube",
    "log_price_ma_5",
    "log_price_ma_10",
    "log_price_ma_20",
    "log_price_ma_60",
    "log_price_ma_220",
]

MACRO_FEATURE_COLS = [
    "sp500_return",
    "sp100_return",
    "nasdaq_return",
    "djia_return",
    "fed_funds_rate",
    "cpi_rate_of_change",
    "recession_indicator",
    "federal_debt_growth",
]

ALL_FEATURE_COLS = TECHNICAL_FEATURE_COLS + MACRO_FEATURE_COLS


def build_macro_and_market_features(
    raw_dir: str = RAW_DIR, trading_dates: pd.DatetimeIndex = None
) -> pd.DataFrame:
    """
    Build aligned macroeconomic and financial market indicators without lookahead bias.
    Publication lags:
    - CPI: 15-day publication lag (mid-month release for previous month's data).
    - NBER Recession: 30-day reporting lag.
    - Federal Debt: 45-day reporting lag (quarterly reports released ~1.5 months post quarter-end).
    - Equity indices and Fed Funds: same-day historical returns/levels.
    """
    logger.info("Building macroeconomic and market indicator features...")
    macro_df = pd.DataFrame(index=trading_dates)
    macro_df.index.name = "DATE"

    # 1. Financial Market Indices
    indices_map = {
        "sp500_return": "yfinance_GSPC.csv",
        "sp100_return": "yfinance_OEX.csv",
        "nasdaq_return": "yfinance_IXIC.csv",
        "djia_return": "yfinance_DJI.csv",
    }
    for col_name, filename in indices_map.items():
        file_path = os.path.join(raw_dir, filename)
        if os.path.exists(file_path):
            df_idx = pd.read_csv(file_path)
            df_idx["DATE"] = pd.to_datetime(df_idx["DATE"])
            df_idx = df_idx.sort_values("DATE").set_index("DATE")
            # Calculate daily log return: ln(Close_t / Close_{t-1})
            close_col = "Close" if "Close" in df_idx.columns else df_idx.columns[0]
            ret_series = np.log(df_idx[close_col] / df_idx[close_col].shift(1))
            # Reindex to trading calendar; fill closed days with 0 return (no price change)
            macro_df[col_name] = ret_series.reindex(trading_dates).fillna(0.0)
        else:
            logger.warning(f"Index file {filename} not found. Filling {col_name} with 0.")
            macro_df[col_name] = 0.0

    # 2. Daily Federal Funds Effective Rate (DFF)
    dff_path = os.path.join(raw_dir, "fred_DFF.csv")
    if os.path.exists(dff_path):
        dff = pd.read_csv(dff_path)
        dff["DATE"] = pd.to_datetime(dff["DATE"])
        dff["DFF"] = pd.to_numeric(dff["DFF"].replace(".", None), errors="coerce")
        dff = dff.dropna().sort_values("DATE").set_index("DATE")
        macro_df["fed_funds_rate"] = dff["DFF"].reindex(trading_dates).ffill()
    else:
        macro_df["fed_funds_rate"] = 0.0

    # 3. Monthly CPI Rate of Change (CPIAUCNS) with 15-day publication lag
    cpi_path = os.path.join(raw_dir, "fred_CPIAUCNS.csv")
    if os.path.exists(cpi_path):
        cpi = pd.read_csv(cpi_path)
        cpi["DATE"] = pd.to_datetime(cpi["DATE"])
        cpi["CPIAUCNS"] = pd.to_numeric(cpi["CPIAUCNS"].replace(".", None), errors="coerce")
        cpi = cpi.dropna().sort_values("DATE").reset_index(drop=True)
        # Rate of change: (CPI_m - CPI_{m-1}) / CPI_{m-1}
        cpi["cpi_change"] = cpi["CPIAUCNS"].pct_change()
        # Conservative publication lag: Available on the 15th of the following month
        cpi["AVAILABLE_DATE"] = cpi["DATE"] + pd.DateOffset(months=1, days=14)
        cpi_avail = cpi.dropna(subset=["cpi_change"]).set_index("AVAILABLE_DATE")["cpi_change"]
        macro_df["cpi_rate_of_change"] = cpi_avail.reindex(trading_dates).ffill()
    else:
        macro_df["cpi_rate_of_change"] = 0.0

    # 4. NBER Recession Indicator (USREC) with 30-day reporting lag
    rec_path = os.path.join(raw_dir, "fred_USREC.csv")
    if os.path.exists(rec_path):
        rec = pd.read_csv(rec_path)
        rec["DATE"] = pd.to_datetime(rec["DATE"])
        rec["USREC"] = pd.to_numeric(rec["USREC"].replace(".", None), errors="coerce")
        rec = rec.dropna().sort_values("DATE").reset_index(drop=True)
        rec["AVAILABLE_DATE"] = rec["DATE"] + pd.DateOffset(months=1)
        rec_avail = rec.set_index("AVAILABLE_DATE")["USREC"]
        macro_df["recession_indicator"] = rec_avail.reindex(trading_dates).ffill()
    else:
        macro_df["recession_indicator"] = 0.0

    # 5. Federal Debt Growth Rate (GFDEBTN) with 45-day reporting lag
    debt_path = os.path.join(raw_dir, "fred_GFDEBTN.csv")
    if os.path.exists(debt_path):
        debt = pd.read_csv(debt_path)
        debt["DATE"] = pd.to_datetime(debt["DATE"])
        debt["GFDEBTN"] = pd.to_numeric(debt["GFDEBTN"].replace(".", None), errors="coerce")
        debt = debt.dropna().sort_values("DATE").reset_index(drop=True)
        debt["debt_growth"] = debt["GFDEBTN"].pct_change()
        # Conservative quarterly reporting lag: 45 days after quarter-end observation
        debt["AVAILABLE_DATE"] = debt["DATE"] + pd.Timedelta(days=45)
        debt_avail = debt.dropna(subset=["debt_growth"]).set_index("AVAILABLE_DATE")["debt_growth"]
        macro_df["federal_debt_growth"] = debt_avail.reindex(trading_dates).ffill()
    else:
        macro_df["federal_debt_growth"] = 0.0

    return macro_df.fillna(0.0)


def build_currency_dataset(
    currency_code: str, raw_dir: str = RAW_DIR, macro_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Construct full feature set and prediction targets for a single currency.
    Features: 15 technical features + 8 macroeconomic features.
    Targets: next-day direction (target_clf) and log return (target_reg).
    """
    cfg = CURRENCY_CONFIG[currency_code]
    file_path = os.path.join(raw_dir, cfg["raw_file"])
    logger.info(f"Processing currency: {currency_code} ({cfg['name']}) from {file_path}...")

    df_raw = pd.read_csv(file_path)
    df_raw["DATE"] = pd.to_datetime(df_raw["DATE"])
    val_col = cfg["val_col"]

    # Clean unobserved holidays ('.' in FRED raw files)
    df_raw[val_col] = pd.to_numeric(df_raw[val_col].replace(".", None), errors="coerce")
    # Drop non-trading holiday dates without interpolation
    df = df_raw.dropna(subset=[val_col]).sort_values("DATE").reset_index(drop=True)
    df.rename(columns={val_col: "rate"}, inplace=True)

    # 1. Base log price and log returns
    df["log_price"] = np.log(df["rate"])
    df["return_t"] = df["log_price"].diff()
    df["return_t_1"] = df["return_t"].shift(1)

    # 2. Moving averages of log returns (5, 10, 20, 60, 120 days)
    for w in [5, 10, 20, 60, 120]:
        df[f"return_ma_{w}"] = df["return_t"].rolling(window=w).mean()

    # 3. Polynomial log price features
    df["log_price_sq"] = df["log_price"] ** 2
    df["log_price_cube"] = df["log_price"] ** 3

    # 4. Moving averages of log prices (5, 10, 20, 60, 220 days)
    for w in [5, 10, 20, 60, 220]:
        df[f"log_price_ma_{w}"] = df["log_price"].rolling(window=w).mean()

    # 5. Future Targets (strictly shifted t+1)
    # Numerical log return target: r_{t+1} = ln(P_{t+1} / P_t) = log_price_{t+1} - log_price_t
    df["target_reg"] = df["log_price"].shift(-1) - df["log_price"]
    # Binary sign prediction target: 1 if P_{t+1} > P_t (up trend), 0 otherwise (down trend)
    df["target_clf"] = (df["target_reg"] > 0).astype(int)
    # Next-day price return ratio: P_{t+1} / P_t = exp(r_{t+1})
    df["target_ratio"] = df["rate"].shift(-1) / df["rate"]

    # 6. Merge macroeconomic features
    if macro_df is not None:
        df = df.merge(macro_df.reset_index(), on="DATE", how="left")

    # 7. Drop rows with uncomputed rolling windows (warmup up to 220 days) and last unshifted target row
    df_clean = df.dropna().reset_index(drop=True)
    df_clean["currency"] = currency_code
    df_clean["quote_convention"] = cfg["quote_convention"]

    logger.info(
        f"Completed feature engineering for {currency_code}: "
        f"{len(df_clean)} rows from {df_clean['DATE'].min().strftime('%Y-%m-%d')} to "
        f"{df_clean['DATE'].max().strftime('%Y-%m-%d')}."
    )
    return df_clean


def partition_dataset(
    df: pd.DataFrame, currency_code: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronologically partitions the dataset into Train, Validation, and Test sets.
    Matches the paper's explicit test window:
      - EUR: 2017-03-03 to 2018-11-30
      - GBP & JPY: 2017-03-07 to 2018-11-30
    Validation set: 440 trading days immediately preceding test start (~10% of dataset).
    Train set: Remaining historical data from ~2001 to validation start (~80% of dataset).
    """
    cfg = CURRENCY_CONFIG[currency_code]
    test_start = pd.to_datetime(cfg["test_start"])
    test_end = pd.to_datetime(cfg["test_end"])

    # Test partition
    test_mask = (df["DATE"] >= test_start) & (df["DATE"] <= test_end)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    # Pre-test data
    pre_test_df = df.loc[df["DATE"] < test_start].copy().reset_index(drop=True)

    # Validation size: 440 days (~10% of total)
    val_len = min(440, int(len(pre_test_df) * 0.11))
    train_df = pre_test_df.iloc[:-val_len].copy().reset_index(drop=True)
    val_df = pre_test_df.iloc[-val_len:].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    logger.info(
        f"[{currency_code}] Splits: Train={len(train_df)} ({train_df['DATE'].min().strftime('%Y-%m-%d')} to {train_df['DATE'].max().strftime('%Y-%m-%d')}) | "
        f"Val={len(val_df)} ({val_df['DATE'].min().strftime('%Y-%m-%d')} to {val_df['DATE'].max().strftime('%Y-%m-%d')}) | "
        f"Test={len(test_df)} ({test_df['DATE'].min().strftime('%Y-%m-%d')} to {test_df['DATE'].max().strftime('%Y-%m-%d')})"
    )
    return train_df, val_df, test_df


def scale_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str] = ALL_FEATURE_COLS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Applies StandardScaler fitted strictly on the training partition only.
    Prevents information leakage from validation and test sets into feature scaling.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    test_scaled = test_df.copy()

    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_scaled, val_scaled, test_scaled, scaler


def run_preprocessing_pipeline(
    raw_dir: str = RAW_DIR,
    processed_dir: str = PROCESSED_DIR,
    figures_dir: str = FIGURES_DIR,
) -> Dict[str, Any]:
    """
    Executes end-to-end preprocessing, feature generation, dataset partitioning,
    and metadata generation for all currency pairs.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Establish common trading calendar from FX series
    sample_fx = pd.read_csv(os.path.join(raw_dir, CURRENCY_CONFIG["GBP"]["raw_file"])).dropna()
    common_dates = pd.to_datetime(sample_fx["DATE"]).sort_values().drop_duplicates()

    # 2. Build macroeconomic features aligned to calendar
    macro_df = build_macro_and_market_features(raw_dir=raw_dir, trading_dates=common_dates)

    summary_metadata = {
        "technical_features_count": len(TECHNICAL_FEATURE_COLS),
        "macro_features_count": len(MACRO_FEATURE_COLS),
        "total_features_count": len(ALL_FEATURE_COLS),
        "technical_features": TECHNICAL_FEATURE_COLS,
        "macro_features": MACRO_FEATURE_COLS,
        "target_definitions": {
            "target_reg": "r_{t+1} = ln(P_{t+1} / P_t): next-day log return",
            "target_clf": "1 if P_{t+1} > P_t else 0: next-day direction",
            "target_ratio": "P_{t+1} / P_t = exp(r_{t+1}): daily return multiplier",
        },
        "currencies": {},
    }

    all_processed_dfs = {}

    for ccy in ["GBP", "EUR", "JPY"]:
        # Build features
        ccy_df = build_currency_dataset(currency_code=ccy, raw_dir=raw_dir, macro_df=macro_df)
        all_processed_dfs[ccy] = ccy_df

        # Chronological partition
        train_df, val_df, test_df = partition_dataset(ccy_df, currency_code=ccy)

        # Scale features using training data only
        train_scaled, val_scaled, test_scaled, scaler = scale_features(train_df, val_df, test_df)

        # Save processed CSVs
        ccy_lower = ccy.lower()
        train_df.to_csv(os.path.join(processed_dir, f"{ccy_lower}_train.csv"), index=False)
        val_df.to_csv(os.path.join(processed_dir, f"{ccy_lower}_val.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, f"{ccy_lower}_test.csv"), index=False)

        train_scaled.to_csv(os.path.join(processed_dir, f"{ccy_lower}_train_scaled.csv"), index=False)
        val_scaled.to_csv(os.path.join(processed_dir, f"{ccy_lower}_val_scaled.csv"), index=False)
        test_scaled.to_csv(os.path.join(processed_dir, f"{ccy_lower}_test_scaled.csv"), index=False)

        # Full combined series with split column
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        combined_df.to_csv(os.path.join(processed_dir, f"{ccy_lower}_full.csv"), index=False)

        # Record metadata
        summary_metadata["currencies"][ccy] = {
            "currency_name": CURRENCY_CONFIG[ccy]["name"],
            "quote_convention": CURRENCY_CONFIG[ccy]["quote_convention"],
            "total_observations": len(combined_df),
            "date_range": [
                combined_df["DATE"].min().strftime("%Y-%m-%d"),
                combined_df["DATE"].max().strftime("%Y-%m-%d"),
            ],
            "train_size": len(train_df),
            "train_range": [
                train_df["DATE"].min().strftime("%Y-%m-%d"),
                train_df["DATE"].max().strftime("%Y-%m-%d"),
            ],
            "val_size": len(val_df),
            "val_range": [
                val_df["DATE"].min().strftime("%Y-%m-%d"),
                val_df["DATE"].max().strftime("%Y-%m-%d"),
            ],
            "test_size": len(test_df),
            "test_range": [
                test_df["DATE"].min().strftime("%Y-%m-%d"),
                test_df["DATE"].max().strftime("%Y-%m-%d"),
            ],
            "train_up_class_ratio": float(train_df["target_clf"].mean()),
            "test_up_class_ratio": float(test_df["target_clf"].mean()),
        }

    # Save metadata JSON
    meta_path = os.path.join(processed_dir, "dataset_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)
    logger.info(f"Processed dataset metadata saved to {meta_path}")

    # Generate exploratory plots
    generate_exploratory_figures(all_processed_dfs, figures_dir=figures_dir)

    return summary_metadata


def generate_exploratory_figures(
    dfs_dict: Dict[str, pd.DataFrame], figures_dir: str = FIGURES_DIR
) -> None:
    """
    Generate exploratory time-series and distribution figures without data leakage.
    Reproduces Figure 1 and visualizes log return distributions.
    """
    logger.info("Generating exploratory analysis plots in results/figures/...")
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Exchange Rate Trends with Partition Boundaries (Reproduction of Figure 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    colors = {"GBP": "#1f77b4", "EUR": "#2ca02c", "JPY": "#d62728"}

    for idx, ccy in enumerate(["GBP", "EUR", "JPY"]):
        ax = axes[idx]
        df = dfs_dict[ccy]
        ax.plot(df["DATE"], df["rate"], label=f"{CURRENCY_CONFIG[ccy]['name']} ({ccy})", color=colors[ccy], lw=1.2)
        ax.set_title(f"{CURRENCY_CONFIG[ccy]['name']} Exchange Rate ({CURRENCY_CONFIG[ccy]['quote_convention']})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Rate", fontsize=10)

        # Highlight test window (March 2017 to Nov 2018)
        test_start = pd.to_datetime(CURRENCY_CONFIG[ccy]["test_start"])
        test_end = pd.to_datetime(CURRENCY_CONFIG[ccy]["test_end"])
        val_start = test_start - pd.Timedelta(days=640)
        ax.axvspan(val_start, test_start, color="orange", alpha=0.15, label="Validation Split (~10%)")
        ax.axvspan(test_start, test_end, color="green", alpha=0.15, label="Test Window (Paper ~10%)")
        ax.legend(loc="upper left", frameon=True, fontsize=9)

    axes[2].set_xlabel("Date", fontsize=11)
    plt.tight_layout()
    trend_path = os.path.join(figures_dir, "exchange_rates_trend.png")
    fig.savefig(trend_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved exchange rate trend figure to {trend_path}")

    # 2. Daily Log Returns Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, ccy in enumerate(["GBP", "EUR", "JPY"]):
        ax = axes[idx]
        df = dfs_dict[ccy]
        returns = df["return_t"] * 100  # in percentage
        sns.histplot(returns, bins=60, kde=True, ax=ax, color=colors[ccy], stat="density")
        mu, sigma = returns.mean(), returns.std()
        ax.set_title(rf"{ccy} Daily Log Returns (%)\n$\mu={mu:.3f}\%,\ \sigma={sigma:.3f}\%$", fontsize=11, fontweight="bold")
        ax.set_xlabel("Daily Log Return (%)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)

    plt.tight_layout()
    dist_path = os.path.join(figures_dir, "log_returns_distribution.png")
    fig.savefig(dist_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved returns distribution figure to {dist_path}")

    # 3. Technical Feature Correlation Heatmap (Train split only to prevent leakage)
    fig, ax = plt.subplots(figsize=(10, 8))
    gbp_train = dfs_dict["GBP"].iloc[:3400]
    corr = gbp_train[TECHNICAL_FEATURE_COLS].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("GBP Technical Features Correlation Matrix (Train Set Only)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    corr_path = os.path.join(figures_dir, "technical_features_correlation.png")
    fig.savefig(corr_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved feature correlation figure to {corr_path}")


if __name__ == "__main__":
    meta = run_preprocessing_pipeline()
    print("\nPreprocessing Pipeline Finished Successfully.")
    print(f"Features: {meta['total_features_count']} ({meta['technical_features_count']} technical + {meta['macro_features_count']} macro)")
    for ccy, details in meta["currencies"].items():
        print(f"  {ccy}: Train={details['train_size']} | Val={details['val_size']} | Test={details['test_size']} (Total: {details['total_observations']})")
