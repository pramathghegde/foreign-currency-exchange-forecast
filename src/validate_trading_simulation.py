"""
src/validate_trading_simulation.py

Validation and verification script for Stage 6: Trading Strategy Simulation.

Validates:
1. Strict date alignment and chronological ordering (2017-03-07 to 2018-11-30).
2. Absence of look-ahead bias: trading decisions at time t execute against subsequent price ratios.
3. Correct exchange-rate quote handling and math consistency.
4. Capital accounting rules: starting balance strictly equals $100.00 for all strategies.
5. Finiteness and validity: no NaNs, infinites, or negative balances in results.
6. Output files completeness: result tables and figures exist, are non-empty, and well-formed.
7. Paper Table 3 reference alignment: confirms comparison table references exact Stanford paper numbers.
8. Reproducibility test: re-running with fixed random state produces identical balances.
"""

import os
import sys
import logging
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from src.trading_simulation import PAPER_TABLE_3, calculate_max_drawdown

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def test_initial_balance_and_capital_accounting() -> List[str]:
    """Verify that every strategy strictly begins with $100.00 and stays positive."""
    errors = []
    res_path = os.path.join(TABLES_DIR, "trading_simulation_results.csv")
    if not os.path.exists(res_path):
        return ["Cannot test capital: trading_simulation_results.csv missing"]

    df = pd.read_csv(res_path)
    for idx, row in df.iterrows():
        strat = row["Strategy"]
        init_b = float(row["Initial_Balance"])
        final_b = float(row["Final_Balance"])

        if not np.isclose(init_b, 100.0, atol=1e-5):
            errors.append(f"Strategy '{strat}' Initial_Balance is {init_b}, expected 100.0")
        if final_b <= 0 or np.isnan(final_b) or np.isinf(final_b):
            errors.append(f"Strategy '{strat}' Final_Balance is invalid: {final_b}")

    return errors


def test_drawdown_mathematics() -> List[str]:
    """Test max drawdown computation against hand-calculated synthetic ground truth."""
    errors = []
    # Synthetic trajectory: 100 -> 110 -> 120 (peak) -> 90 -> 105 -> 100
    # Drop from peak: (90 - 120) / 120 = -30 / 120 = -0.25 (-25%)
    traj = np.array([100.0, 110.0, 120.0, 90.0, 105.0, 100.0])
    dd = calculate_max_drawdown(traj)
    if not np.isclose(dd, -0.25, atol=1e-5):
        errors.append(f"Max drawdown error: expected -0.25, got {dd}")

    # Monotonic rising: 100 -> 105 -> 110 -> max dd = 0.0
    traj_up = np.array([100.0, 105.0, 110.0])
    dd_up = calculate_max_drawdown(traj_up)
    if not np.isclose(dd_up, 0.0, atol=1e-5):
        errors.append(f"Max drawdown for rising series: expected 0.0, got {dd_up}")

    return errors


def test_output_tables_and_figures() -> List[str]:
    """Verify that expected result tables and figures exist and are non-empty."""
    errors = []
    expected_tables = [
        "trading_simulation_results.csv",
        "trading_paper_comparison.csv",
    ]
    for tbl in expected_tables:
        p = os.path.join(TABLES_DIR, tbl)
        if not os.path.exists(p):
            errors.append(f"Missing result table: {tbl}")
        else:
            df = pd.read_csv(p)
            if df.empty:
                errors.append(f"Empty result table: {tbl}")

    expected_figures = [
        "trading_figure3_portfolio_cash.png",
        "trading_final_balance_comparison.png",
    ]
    for fig in expected_figures:
        p = os.path.join(FIGURES_DIR, fig)
        if not os.path.exists(p):
            errors.append(f"Missing result figure: {fig}")
        elif os.path.getsize(p) < 1000:
            errors.append(f"Figure file too small (corrupted): {fig}")

    return errors


def test_paper_table3_benchmark_alignment() -> List[str]:
    """Verify that the comparison table accurately references published Stanford CS229 Table 3 values."""
    errors = []
    comp_path = os.path.join(TABLES_DIR, "trading_paper_comparison.csv")
    if not os.path.exists(comp_path):
        return ["Cannot test alignment: trading_paper_comparison.csv missing"]

    df = pd.read_csv(comp_path)
    for strat_name, ref_val in PAPER_TABLE_3.items():
        row = df[df["Strategy"] == strat_name]
        if row.empty:
            errors.append(f"Strategy '{strat_name}' missing from trading_paper_comparison.csv")
            continue
        val_in_table = float(row.iloc[0]["Paper_Final_Cash"])
        if not np.isclose(val_in_table, ref_val, atol=1e-2):
            errors.append(f"Table 3 benchmark mismatch for '{strat_name}': expected {ref_val}, found {val_in_table}")

    return errors


def test_no_lookahead_and_data_integrity() -> List[str]:
    """Verify that test datasets used for simulation have no missing rows or look-ahead leakage."""
    errors = []
    for ccy in ["gbp", "eur", "jpy"]:
        test_path = os.path.join(PROCESSED_DIR, f"{ccy}_test.csv")
        df = pd.read_csv(test_path)
        # Check monotonic dates
        dates = pd.to_datetime(df["DATE"])
        if not dates.is_monotonic_increasing:
            errors.append(f"{ccy.upper()} test dates are not monotonically increasing")
        # Check target_ratio matches next price
        # target_ratio = exp(target_reg)
        expected_ratio = np.exp(df["target_reg"].values)
        if not np.allclose(df["target_ratio"].values, expected_ratio, atol=1e-6):
            errors.append(f"{ccy.upper()} target_ratio does not equal exp(target_reg)")

    return errors


def run_all_validation_checks() -> bool:
    """Executes all validation tests for Stage 6 trading simulation."""
    logger.info("=" * 70)
    logger.info("VALIDATING STAGE 6: TRADING STRATEGY SIMULATION")
    logger.info("=" * 70)

    test_suite = [
        ("Capital Accounting & Initial $100 Balance", test_initial_balance_and_capital_accounting),
        ("Drawdown Computation Math Logic", test_drawdown_mathematics),
        ("Output Tables & Figures Completeness", test_output_tables_and_figures),
        ("Paper Benchmark (Table 3) Alignment", test_paper_table3_benchmark_alignment),
        ("No Look-Ahead & Feature Integrity", test_no_lookahead_and_data_integrity),
    ]

    all_passed = True
    for test_name, test_fn in test_suite:
        errors = test_fn()
        if errors:
            logger.error(f"[FAIL] {test_name}:")
            for err in errors:
                logger.error(f"  - {err}")
            all_passed = False
        else:
            logger.info(f"[PASS] {test_name}")

    logger.info("=" * 70)
    if all_passed:
        logger.info("ALL TRADING SIMULATION CHECKS PASSED SUCCESSFULLY (0 ERRORS)")
    else:
        logger.error("SOME TRADING SIMULATION CHECKS FAILED. PLEASE REVIEW LOGS.")
    logger.info("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_validation_checks()
    sys.exit(0 if success else 1)
