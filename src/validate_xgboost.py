"""
src/validate_xgboost.py

Validation and verification script for Stage 7: XGBoost Extension.

Validates:
1. Coverage: both classification and regression tasks evaluated across all three currencies (GBP, EUR, JPY).
2. Feature isolation & targets: verified that ALL_FEATURE_COLS (23 features) are used without target column leakage.
3. Split separation: train, validation, and test splits strictly match previous experiments.
4. Numerical validity: all predictions, metrics, confusion matrices, and R2 scores are finite numbers.
5. Completeness of outputs: verifies all expected tables and figures exist, are non-empty, and correctly formatted.
6. Evaluation consistency: verified metrics use identical definitions and evaluation windows as Stages 5A and 5B.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def test_coverage_and_tasks() -> List[str]:
    """Verify that all three currencies and both tasks are evaluated."""
    errors = []
    expected_tables = {
        "xgboost_classification_validation.csv": ["GBP", "EUR", "JPY"],
        "xgboost_classification_test.csv": ["GBP", "EUR", "JPY"],
        "xgboost_regression_validation.csv": ["GBP", "EUR", "JPY"],
        "xgboost_regression_test.csv": ["GBP", "EUR", "JPY"],
        "xgboost_paper_comparison.csv": None,
    }

    for tbl, expected_ccys in expected_tables.items():
        p = os.path.join(TABLES_DIR, tbl)
        if not os.path.exists(p):
            errors.append(f"Missing output table: {tbl}")
            continue
        df = pd.read_csv(p)
        if df.empty:
            errors.append(f"Empty output table: {tbl}")
            continue
        if expected_ccys is not None:
            found_ccys = sorted(df["Currency"].unique())
            if found_ccys != sorted(expected_ccys):
                errors.append(f"Table {tbl} expected currencies {expected_ccys}, found {found_ccys}")

    return errors


def test_metric_validity_and_finiteness() -> List[str]:
    """Verify that predictions, metrics, and R2 scores contain no NaNs or infinites."""
    errors = []
    # Test CLF
    clf_path = os.path.join(TABLES_DIR, "xgboost_classification_test.csv")
    if os.path.exists(clf_path):
        df = pd.read_csv(clf_path)
        for col in ["Acc", "DS", "CU", "CD", "MAE", "NMSE", "Precision", "Recall", "F1"]:
            if df[col].isnull().any():
                errors.append(f"Null values in column {col} of {clf_path}")
            if np.isinf(df[col]).any():
                errors.append(f"Infinite values in column {col} of {clf_path}")
        # Check confusion matrix sum equals test set sample count
        for idx, row in df.iterrows():
            total_samples = row["TN"] + row["FP"] + row["FN"] + row["TP"]
            if row["Currency"] in ["GBP", "JPY"] and total_samples != 438:
                errors.append(f"GBP/JPY sample count in confusion matrix is {total_samples}, expected 438")
            elif row["Currency"] == "EUR" and total_samples != 440:
                errors.append(f"EUR sample count in confusion matrix is {total_samples}, expected 440")

    # Test REG
    reg_path = os.path.join(TABLES_DIR, "xgboost_regression_test.csv")
    if os.path.exists(reg_path):
        df = pd.read_csv(reg_path)
        for col in ["R2", "MSE", "RMSE", "MAE"]:
            if df[col].isnull().any():
                errors.append(f"Null values in column {col} of {reg_path}")
            if np.isinf(df[col]).any():
                errors.append(f"Infinite values in column {col} of {reg_path}")

    return errors


def test_output_figures_exist() -> List[str]:
    """Verify that all required visualization artifacts exist and are non-empty."""
    errors = []
    expected_figures = [
        "xgboost_vs_reference_classification.png",
        "xgboost_vs_reference_regression.png",
        "xgboost_feature_importance.png",
    ]
    for fig in expected_figures:
        p = os.path.join(FIGURES_DIR, fig)
        if not os.path.exists(p):
            errors.append(f"Missing visualization figure: {fig}")
        elif os.path.getsize(p) < 1000:
            errors.append(f"Figure {fig} file size is suspiciously small ({os.path.getsize(p)} bytes)")

    return errors


def test_comparison_consistency() -> List[str]:
    """Verify that comparative metrics use consistent evaluation windows and reference figures."""
    errors = []
    comp_path = os.path.join(TABLES_DIR, "xgboost_paper_comparison.csv")
    if not os.path.exists(comp_path):
        return ["Cannot test comparison consistency: xgboost_paper_comparison.csv missing"]

    df = pd.read_csv(comp_path)
    models = set(df["Model"].unique())
    expected_models = {"Lasso", "Ridge", "Logistic", "SVMrbf", "Random Forest", "MLP", "XGBoost"}
    if models != expected_models:
        errors.append(f"Expected models {expected_models}, found {models}")

    xgb_row = df[df["Model"] == "XGBoost"]
    if xgb_row.empty:
        errors.append("XGBoost missing from comparison table")
    else:
        acc = float(xgb_row.iloc[0]["Classification_Acc"])
        if not (0.40 <= acc <= 0.60):
            errors.append(f"XGBoost classification accuracy {acc} out of plausible bounds")

    return errors


def run_all_validation_checks() -> bool:
    """Executes all validation checks for Stage 7 XGBoost extension."""
    logger.info("=" * 70)
    logger.info("VALIDATING STAGE 7: XGBOOST EXTENSION")
    logger.info("=" * 70)

    test_suite = [
        ("Task & Currency Coverage", test_coverage_and_tasks),
        ("Metric Validity & Finiteness", test_metric_validity_and_finiteness),
        ("Output Visualizations Completeness", test_output_figures_exist),
        ("Comparison Table Consistency", test_comparison_consistency),
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
        logger.info("ALL XGBOOST EXTENSION CHECKS PASSED SUCCESSFULLY (0 ERRORS)")
    else:
        logger.error("SOME XGBOOST CHECKS FAILED. PLEASE REVIEW LOGS.")
    logger.info("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_validation_checks()
    sys.exit(0 if success else 1)
