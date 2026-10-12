"""
src/xgboost_models.py

Stage 7: XGBoost Extension for Foreign Currency Exchange Rate Forecast.
Implements the proposed additional machine learning model (XGBoost) not present in the
original Stanford CS229 (Spring 2019) paper:
1. XGBoost Classifier (Sign Prediction / Directional Classification)
2. XGBoost Regressor (Continuous Next-Day Return Regression)

Protocol:
- Evaluates GBP, EUR, and JPY independently using identical chronological splits
  (Train: 2001-2015, Validation: 2015-2017, Test: 2017-2018).
- Reuses ALL_FEATURE_COLS (23 features: 15 technical indicators + 8 financial/macro indicators).
- Tunes hyperparameters strictly on the validation set (no test data leakage).
- Evaluates test set out-of-sample only after model selection.
- Computes comprehensive classification and regression metrics matching Experiments 1 & 2.
- Generates cross-model benchmark comparison tables and figures:
  * Accuracy comparison bar chart
  * Regression test R2 bar chart
  * Feature importance bar chart
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

from src.feature_engineering import ALL_FEATURE_COLS
from src.classification import compute_classification_metrics
from src.regression import compute_regression_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")


def load_currency_data(currency_code: str) -> Dict[str, pd.DataFrame]:
    """Loads raw unscaled train, val, test CSVs for a currency."""
    ccy = currency_code.lower()
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ccy}_train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ccy}_val.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ccy}_test.csv"))
    return {"train": train_df, "val": val_df, "test": test_df}


def tune_and_train_xgboost_classifier(
    data: Dict[str, pd.DataFrame], random_state: int = 42
) -> Tuple[XGBClassifier, Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]:
    """
    Performs validation-based grid tuning for XGBoost Classifier on direction target ('target_clf').
    """
    X_train = data["train"][ALL_FEATURE_COLS]
    y_train = data["train"]["target_clf"].values
    X_val = data["val"][ALL_FEATURE_COLS]
    y_val = data["val"]["target_clf"].values
    X_test = data["test"][ALL_FEATURE_COLS]
    y_test = data["test"]["target_clf"].values

    param_grid = [
        {"max_depth": 2, "learning_rate": 0.03, "subsample": 0.7, "colsample": 0.8, "alpha": 0.1},
        {"max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample": 0.8, "alpha": 0.1},
        {"max_depth": 3, "learning_rate": 0.03, "subsample": 0.7, "colsample": 0.8, "alpha": 0.1},
        {"max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample": 0.8, "alpha": 0.1},
        {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.7, "colsample": 0.8, "alpha": 0.1},
        {"max_depth": 5, "learning_rate": 0.05, "subsample": 0.7, "colsample": 0.8, "alpha": 0.1},
        {"max_depth": 5, "learning_rate": 0.10, "subsample": 0.7, "colsample": 0.8, "alpha": 0.1},
    ]

    best_score = -1.0
    best_params = None
    best_model = None

    for p in param_grid:
        clf = XGBClassifier(
            n_estimators=150,
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=p["subsample"],
            colsample_bytree=p["colsample"],
            reg_alpha=p["alpha"],
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        if val_acc > best_score:
            best_score = val_acc
            best_params = p
            best_model = clf

    # Compute validation and test metrics
    val_preds = best_model.predict(X_val)
    test_preds = best_model.predict(X_test)

    val_metrics = compute_classification_metrics(y_val, val_preds)
    test_metrics = compute_classification_metrics(y_test, test_preds)

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    test_metrics["TN"] = int(cm[0, 0])
    test_metrics["FP"] = int(cm[0, 1])
    test_metrics["FN"] = int(cm[1, 0])
    test_metrics["TP"] = int(cm[1, 1])

    return best_model, best_params, val_metrics, test_metrics, test_preds


def tune_and_train_xgboost_regressor(
    data: Dict[str, pd.DataFrame], random_state: int = 42
) -> Tuple[XGBRegressor, Dict[str, Any], Dict[str, Any], Dict[str, Any], np.ndarray]:
    """
    Performs validation-based grid tuning for XGBoost Regressor on log return target ('target_reg').
    """
    X_train = data["train"][ALL_FEATURE_COLS]
    y_train = data["train"]["target_reg"].values
    X_val = data["val"][ALL_FEATURE_COLS]
    y_val = data["val"]["target_reg"].values
    X_test = data["test"][ALL_FEATURE_COLS]
    y_test = data["test"]["target_reg"].values

    param_grid = [
        {"max_depth": 2, "learning_rate": 0.01, "alpha": 0.1},
        {"max_depth": 2, "learning_rate": 0.03, "alpha": 0.1},
        {"max_depth": 2, "learning_rate": 0.05, "alpha": 0.01},
        {"max_depth": 3, "learning_rate": 0.01, "alpha": 0.1},
        {"max_depth": 3, "learning_rate": 0.03, "alpha": 0.1},
        {"max_depth": 3, "learning_rate": 0.05, "alpha": 0.1},
        {"max_depth": 5, "learning_rate": 0.01, "alpha": 1.0},
        {"max_depth": 5, "learning_rate": 0.03, "alpha": 0.1},
    ]

    best_score = -np.inf
    best_params = None
    best_model = None

    for p in param_grid:
        reg = XGBRegressor(
            n_estimators=150,
            max_depth=p["max_depth"],
            learning_rate=p["learning_rate"],
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=p["alpha"],
            reg_lambda=1.0,
            random_state=random_state,
        )
        reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_r2 = r2_score(y_val, reg.predict(X_val))
        if val_r2 > best_score:
            best_score = val_r2
            best_params = p
            best_model = reg

    val_preds = best_model.predict(X_val)
    test_preds = best_model.predict(X_test)

    val_metrics = compute_regression_metrics(y_val, val_preds)
    test_metrics = compute_regression_metrics(y_test, test_preds)

    return best_model, best_params, val_metrics, test_metrics, test_preds


def plot_xgboost_evaluations(
    clf_test_df: pd.DataFrame,
    reg_test_df: pd.DataFrame,
    feature_importances: Dict[str, pd.Series],
) -> None:
    """Generates comparison plots and feature importance figures."""
    # 1. Classification Accuracy Comparison: XGBoost vs Reference Models
    clf_ref_path = os.path.join(TABLES_DIR, "classification_test_results.csv")
    if os.path.exists(clf_ref_path):
        ref_clf = pd.read_csv(clf_ref_path)
        combined_clf = pd.concat([
            ref_clf[["Currency", "Model", "Acc"]],
            clf_test_df[["Currency", "Model", "Acc"]],
        ], ignore_index=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=combined_clf, x="Currency", y="Acc", hue="Model", palette="tab10")
        plt.axhline(0.50, color="gray", linestyle="--", alpha=0.7, label="Random Guess (50%)")
        plt.title("Classification Accuracy Comparison: XGBoost vs. Reference Models", fontsize=12, fontweight="bold")
        plt.ylabel("Test Set Accuracy", fontsize=10)
        plt.ylim(0.40, 0.60)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "xgboost_vs_reference_classification.png"), dpi=300)
        plt.close()

    # 2. Regression R2 Comparison: XGBoost vs Reference Models (GBP, EUR, JPY)
    reg_ref_path = os.path.join(TABLES_DIR, "regression_test_results.csv")
    if os.path.exists(reg_ref_path):
        ref_reg = pd.read_csv(reg_ref_path)
        # Exclude extreme outlier RF GBP to view scale nicely, or clamp
        combined_reg = pd.concat([
            ref_reg[["Currency", "Model", "R2"]],
            reg_test_df[["Currency", "Model", "R2"]],
        ], ignore_index=True)

        plt.figure(figsize=(11, 6))
        # Filter models for clear visibility
        filtered_reg = combined_reg[combined_reg["Model"].isin(["XGBoost", "Lasso", "Ridge", "SVM poly", "MLP", "SVM rbf"])].copy()
        sns.barplot(data=filtered_reg, x="Currency", y="R2", hue="Model", palette="Set2")
        plt.axhline(0.0, color="black", linestyle="--", alpha=0.7)
        plt.title("Regression Test R² Comparison: XGBoost vs. Linear / SVR / MLP Models", fontsize=12, fontweight="bold")
        plt.ylabel("Test R² Score", fontsize=10)
        plt.ylim(-0.15, 0.05)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "xgboost_vs_reference_regression.png"), dpi=300)
        plt.close()

    # 3. XGBoost Feature Importance Plot (Average across currencies)
    avg_importance = pd.DataFrame(feature_importances).mean(axis=1).sort_values(ascending=True)

    plt.figure(figsize=(10, 8))
    avg_importance.plot(kind="barh", color="#1f77b4")
    plt.title("XGBoost Extension: Average Feature Importance (Gain / Weight)", fontsize=12, fontweight="bold")
    plt.xlabel("Importance Score", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "xgboost_feature_importance.png"), dpi=300)
    plt.close()


def run_xgboost_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes Stage 7 XGBoost experiments across GBP, EUR, and JPY.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("=" * 75)
    logger.info("STAGE 7: XGBOOST EXTENSION EXPERIMENTS")
    logger.info("=" * 75)

    currencies = ["GBP", "EUR", "JPY"]
    clf_val_records = []
    clf_test_records = []
    reg_val_records = []
    reg_test_records = []
    feature_importances = {}

    for ccy in currencies:
        logger.info(f"\n--- Training & Tuning XGBoost for {ccy} ---")
        data = load_currency_data(ccy)

        # 1. XGBoost Classification
        clf_model, clf_params, clf_val_m, clf_test_m, _ = tune_and_train_xgboost_classifier(data, random_state=42)
        clf_cfg_str = f"max_depth={clf_params['max_depth']}, lr={clf_params['learning_rate']}, sub={clf_params['subsample']}"

        clf_val_rec = {"Currency": ccy, "Model": "XGBoost", "Config": clf_cfg_str}
        clf_val_rec.update(clf_val_m)
        clf_val_records.append(clf_val_rec)

        clf_test_rec = {"Currency": ccy, "Model": "XGBoost", "Config": clf_cfg_str}
        clf_test_rec.update(clf_test_m)
        clf_test_records.append(clf_test_rec)

        logger.info(f"[{ccy} TEST CLF] Acc={clf_test_m['Acc']:.4f} | DS={clf_test_m['DS']:.4f} | CU={clf_test_m['CU']:.4f} | CD={clf_test_m['CD']:.4f} | Config: {clf_cfg_str}")

        # 2. XGBoost Regression
        reg_model, reg_params, reg_val_m, reg_test_m, _ = tune_and_train_xgboost_regressor(data, random_state=42)
        reg_cfg_str = f"max_depth={reg_params['max_depth']}, lr={reg_params['learning_rate']}, alpha={reg_params['alpha']}"

        reg_val_rec = {"Currency": ccy, "Model": "XGBoost", "Config": reg_cfg_str}
        reg_val_rec.update(reg_val_m)
        reg_val_records.append(reg_val_rec)

        reg_test_rec = {"Currency": ccy, "Model": "XGBoost", "Config": reg_cfg_str}
        reg_test_rec.update(reg_test_m)
        reg_test_records.append(reg_test_rec)

        logger.info(f"[{ccy} TEST REG] R2={reg_test_m['R2']:.4f} | MSE={reg_test_m['MSE']:.8f} | RMSE={reg_test_m['RMSE']:.6f} | Config: {reg_cfg_str}")

        # Save feature importance
        feature_importances[ccy] = pd.Series(clf_model.feature_importances_, index=ALL_FEATURE_COLS)

    clf_val_df = pd.DataFrame(clf_val_records)
    clf_test_df = pd.DataFrame(clf_test_records)
    reg_val_df = pd.DataFrame(reg_val_records)
    reg_test_df = pd.DataFrame(reg_test_records)

    # Save detailed CSVs
    clf_val_path = os.path.join(TABLES_DIR, "xgboost_classification_validation.csv")
    clf_test_path = os.path.join(TABLES_DIR, "xgboost_classification_test.csv")
    reg_val_path = os.path.join(TABLES_DIR, "xgboost_regression_validation.csv")
    reg_test_path = os.path.join(TABLES_DIR, "xgboost_regression_test.csv")

    clf_val_df.to_csv(clf_val_path, index=False)
    clf_test_df.to_csv(clf_test_path, index=False)
    reg_val_df.to_csv(reg_val_path, index=False)
    reg_test_df.to_csv(reg_test_path, index=False)

    logger.info(f"Saved {clf_val_path}")
    logger.info(f"Saved {clf_test_path}")
    logger.info(f"Saved {reg_val_path}")
    logger.info(f"Saved {reg_test_path}")

    # Create Overall Synthesis Comparison Table against Reference Models & Paper
    # Classification averages
    clf_avg = clf_test_df[["Acc", "NMSE", "MAE", "DS", "CU", "CD", "Precision", "Recall", "F1"]].mean().round(4)
    reg_avg = reg_test_df[["R2", "MSE", "RMSE", "MAE"]].mean().round(6)

    comp_records = []
    # Load reference tables
    clf_ref = pd.read_csv(os.path.join(TABLES_DIR, "classification_paper_comparison.csv"))
    reg_ref = pd.read_csv(os.path.join(TABLES_DIR, "regression_paper_comparison.csv"))

    for _, row in clf_ref.iterrows():
        m_name = row["Model"]
        reg_match = reg_ref[reg_ref["Model"] == m_name]
        reg_r2 = reg_match["Our_R2"].iloc[0] if not reg_match.empty else "N/A"
        comp_records.append({
            "Model": m_name,
            "Type": "Paper Reference",
            "Classification_Acc": row["Our_Acc"],
            "Paper_Table1_Acc": row["Paper_Acc"],
            "Regression_R2": reg_r2,
            "CU_Up": row["Our_CU"],
            "CD_Down": row["Our_CD"],
        })

    comp_records.append({
        "Model": "XGBoost",
        "Type": "New Extension Model",
        "Classification_Acc": clf_avg["Acc"],
        "Paper_Table1_Acc": "N/A",
        "Regression_R2": round(float(reg_avg["R2"]), 4),
        "CU_Up": clf_avg["CU"],
        "CD_Down": clf_avg["CD"],
    })

    comp_df = pd.DataFrame(comp_records)
    comp_path = os.path.join(TABLES_DIR, "xgboost_paper_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved {comp_path}")

    # Generate plots
    plot_xgboost_evaluations(clf_test_df, reg_test_df, feature_importances)

    return clf_val_df, clf_test_df, reg_val_df, reg_test_df, comp_df


if __name__ == "__main__":
    run_xgboost_experiments()
