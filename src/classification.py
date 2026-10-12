"""
src/classification.py

Experiment 1: Sign Prediction (Classification) for Foreign Currency Exchange Rate Forecast.
Replicates and benchmarks the 6 machine learning techniques reported in Section 5.1 & Table 1
of the Stanford CS229 (Spring 2019) research paper:
1. Logistic Regression
2. Lasso (L1-regularized classification)
3. Ridge (L2-regularized linear classification)
4. Support Vector Machine (SVM with RBF kernel)
5. Random Forest (1000 trees, max depth 100 per paper specification)
6. Multi-Layer Perceptron (MLP with 2 hidden layers per paper specification)

Protocol:
- Evaluates GBP, EUR, and JPY independently using chronological train, validation, and test splits.
- Scaled features used for linear models, SVM, and MLP; unscaled features for Random Forest.
- Hyperparameter tuning conducted strictly on train/validation sets (no test data leakage).
- Evaluates test set out-of-sample only after model selection.
- Computes Accuracy, NMSE, MAE, Directional Symmetry (DS), Correct Up (CU), Correct Down (CD),
  Precision, Recall, and F1-score.
"""

import os
import json
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

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.feature_engineering import ALL_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Paper Table 1 Reference Values (3-currency averages reported in Stanford CS229 2019)
PAPER_TABLE_1 = {
    "Lasso": {"Acc": 0.52, "NMSE": 1.90, "MAE": 0.95, "DS": 0.52, "CU": 0.47, "CD": 0.58},
    "Ridge": {"Acc": 0.55, "NMSE": 1.81, "MAE": 0.91, "DS": 0.55, "CU": 0.58, "CD": 0.51},
    "Logistic": {"Acc": 0.51, "NMSE": 1.97, "MAE": 0.98, "DS": 0.51, "CU": 0.45, "CD": 0.57},
    "SVMrbf": {"Acc": 0.51, "NMSE": 1.95, "MAE": 0.97, "DS": 0.51, "CU": 0.57, "CD": 0.52},
    "Random Forest": {"Acc": 0.53, "NMSE": 2.03, "MAE": 1.02, "DS": 0.49, "CU": 0.44, "CD": 0.49},
    "MLP": {"Acc": 0.52, "NMSE": 1.92, "MAE": 0.96, "DS": 0.52, "CU": 0.54, "CD": 0.52},
}


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes all classification metrics reported in Table 1 of the Stanford paper,
    plus standard classification diagnostics (precision, recall, f1).

    Metrics:
    - Accuracy: standard fraction of correct predictions.
    - Directional Symmetry (DS): fraction of predictions with matching direction sign.
    - Correct Up (CU): Recall of the Up class (y=1).
    - Correct Down (CD): Specificity / Recall of the Down class (y=0).
    - MAE (paper formulation): Mean Absolute Error on directional scale {-1, +1}:
        MAE = 2 * (1 - Accuracy).
    - NMSE (paper formulation): Normalized MSE on directional scale {-1, +1}:
        NMSE = 4 * (1 - Accuracy) / Var(y_{-1, +1}).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    ds = acc  # Binary {0, 1} sign prediction directional symmetry equals accuracy

    # Up trend metrics (y=1)
    up_total = (y_true == 1).sum()
    up_correct = ((y_true == 1) & (y_pred == 1)).sum()
    cu = float(up_correct / up_total) if up_total > 0 else 0.0

    # Down trend metrics (y=0)
    down_total = (y_true == 0).sum()
    down_correct = ((y_true == 0) & (y_pred == 0)).sum()
    cd = float(down_correct / down_total) if down_total > 0 else 0.0

    # Paper {-1, +1} directional loss metrics
    y_true_pm = np.where(y_true == 1, 1.0, -1.0)
    y_pred_pm = np.where(y_pred == 1, 1.0, -1.0)
    mae_paper = float(np.mean(np.abs(y_true_pm - y_pred_pm)))
    mse_paper = float(np.mean((y_true_pm - y_pred_pm) ** 2))
    var_y_pm = float(np.var(y_true_pm))
    nmse_paper = float(mse_paper / var_y_pm) if var_y_pm > 1e-8 else mse_paper

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "Acc": round(acc, 4),
        "DS": round(ds, 4),
        "CU": round(cu, 4),
        "CD": round(cd, 4),
        "MAE": round(mae_paper, 4),
        "NMSE": round(nmse_paper, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
    }


def load_split_data(
    currency_code: str, processed_dir: str = PROCESSED_DIR
) -> Dict[str, Dict[str, Any]]:
    """
    Loads train, validation, and test datasets for a given currency.
    Returns both unscaled (for RF) and scaled (for linear/SVM/MLP) representations.
    """
    ccy = currency_code.lower()
    train_raw = pd.read_csv(os.path.join(processed_dir, f"{ccy}_train.csv"))
    val_raw = pd.read_csv(os.path.join(processed_dir, f"{ccy}_val.csv"))
    test_raw = pd.read_csv(os.path.join(processed_dir, f"{ccy}_test.csv"))

    train_sc = pd.read_csv(os.path.join(processed_dir, f"{ccy}_train_scaled.csv"))
    val_sc = pd.read_csv(os.path.join(processed_dir, f"{ccy}_val_scaled.csv"))
    test_sc = pd.read_csv(os.path.join(processed_dir, f"{ccy}_test_scaled.csv"))

    return {
        "unscaled": {
            "X_train": train_raw[ALL_FEATURE_COLS],
            "y_train": train_raw["target_clf"].values,
            "X_val": val_raw[ALL_FEATURE_COLS],
            "y_val": val_raw["target_clf"].values,
            "X_test": test_raw[ALL_FEATURE_COLS],
            "y_test": test_raw["target_clf"].values,
            "dates_test": test_raw["DATE"].values,
        },
        "scaled": {
            "X_train": train_sc[ALL_FEATURE_COLS],
            "y_train": train_sc["target_clf"].values,
            "X_val": val_sc[ALL_FEATURE_COLS],
            "y_val": val_sc["target_clf"].values,
            "X_test": test_sc[ALL_FEATURE_COLS],
            "y_test": test_sc["target_clf"].values,
            "dates_test": test_sc["DATE"].values,
        },
    }


def tune_and_train_models(
    currency_data: Dict[str, Dict[str, Any]], random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Trains candidate model configurations on training data, validates on validation data,
    selects best hyperparameter configurations, and evaluates on out-of-sample test data.
    """
    unscaled = currency_data["unscaled"]
    scaled = currency_data["scaled"]

    y_val = scaled["y_val"]
    y_test = scaled["y_test"]

    best_models = {}
    val_results = {}
    test_results = {}
    predictions_test = {}

    # 1. Logistic Regression (tuning C on validation)
    best_c, best_score, best_lr = 1.0, -1.0, None
    for C_cand in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        model = LogisticRegression(C=C_cand, max_iter=1000, random_state=random_state)
        model.fit(scaled["X_train"], scaled["y_train"])
        score = accuracy_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score:
            best_score, best_c, best_lr = score, C_cand, model
    best_models["Logistic"] = {"model": best_lr, "config": f"C={best_c}", "scaled": True}
    val_results["Logistic"] = compute_classification_metrics(y_val, best_lr.predict(scaled["X_val"]))

    # 2. Lasso: L1-regularized classification
    # Documented method: Logistic regression with L1 penalty via liblinear solver
    best_c_l1, best_score_l1, best_lasso = 1.0, -1.0, None
    for C_cand in [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=C_cand, max_iter=1000, random_state=random_state
        )
        model.fit(scaled["X_train"], scaled["y_train"])
        score = accuracy_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_l1:
            best_score_l1, best_c_l1, best_lasso = score, C_cand, model
    best_models["Lasso"] = {"model": best_lasso, "config": f"penalty=l1, C={best_c_l1}", "scaled": True}
    val_results["Lasso"] = compute_classification_metrics(y_val, best_lasso.predict(scaled["X_val"]))

    # 3. Ridge: L2-regularized linear classification via RidgeClassifier
    # Documented method: RidgeClassifier minimizes penalized MSE on {-1, +1} with decision boundary 0
    best_alpha, best_score_rc, best_ridge = 1.0, -1.0, None
    for alpha_cand in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]:
        model = RidgeClassifier(alpha=alpha_cand, random_state=random_state)
        model.fit(scaled["X_train"], scaled["y_train"])
        score = accuracy_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_rc:
            best_score_rc, best_alpha, best_ridge = score, alpha_cand, model
    best_models["Ridge"] = {"model": best_ridge, "config": f"alpha={best_alpha}", "scaled": True}
    val_results["Ridge"] = compute_classification_metrics(y_val, best_ridge.predict(scaled["X_val"]))

    # 4. Support Vector Machine (RBF Kernel)
    best_svm_cfg, best_score_svm, best_svm = None, -1.0, None
    for C_cand in [0.1, 0.5, 1.0, 2.0]:
        for gamma_cand in ["scale", 0.01, 0.05]:
            model = SVC(kernel="rbf", C=C_cand, gamma=gamma_cand, random_state=random_state)
            model.fit(scaled["X_train"], scaled["y_train"])
            score = accuracy_score(y_val, model.predict(scaled["X_val"]))
            if score > best_score_svm:
                best_score_svm = score
                best_svm_cfg = f"C={C_cand}, gamma={gamma_cand}"
                best_svm = model
    best_models["SVMrbf"] = {"model": best_svm, "config": best_svm_cfg, "scaled": True}
    val_results["SVMrbf"] = compute_classification_metrics(y_val, best_svm.predict(scaled["X_val"]))

    # 5. Random Forest (fixed to paper specification: 1000 trees, max depth 100, unscaled data)
    rf = RandomForestClassifier(
        n_estimators=1000, max_depth=100, random_state=random_state, n_jobs=-1
    )
    rf.fit(unscaled["X_train"], unscaled["y_train"])
    best_models["Random Forest"] = {
        "model": rf,
        "config": "n_estimators=1000, max_depth=100",
        "scaled": False,
    }
    val_results["Random Forest"] = compute_classification_metrics(
        unscaled["y_val"], rf.predict(unscaled["X_val"])
    )

    # 6. Multi-Layer Perceptron (MLP with 2 hidden layers per Section 4.3 equation (1))
    best_mlp_cfg, best_score_mlp, best_mlp = None, -1.0, None
    for hidden_cand in [(64, 32), (100, 50), (32, 16)]:
        for alpha_cand in [0.0001, 0.001, 0.01]:
            model = MLPClassifier(
                hidden_layer_sizes=hidden_cand,
                alpha=alpha_cand,
                max_iter=500,
                random_state=random_state,
                early_stopping=True,
            )
            model.fit(scaled["X_train"], scaled["y_train"])
            score = accuracy_score(y_val, model.predict(scaled["X_val"]))
            if score > best_score_mlp:
                best_score_mlp = score
                best_mlp_cfg = f"hidden={hidden_cand}, alpha={alpha_cand}"
                best_mlp = model
    best_models["MLP"] = {"model": best_mlp, "config": best_mlp_cfg, "scaled": True}
    val_results["MLP"] = compute_classification_metrics(y_val, best_mlp.predict(scaled["X_val"]))

    # Out-of-sample Test Evaluation for all selected models
    for name, item in best_models.items():
        model = item["model"]
        X_test = scaled["X_test"] if item["scaled"] else unscaled["X_test"]
        y_test_curr = y_test if item["scaled"] else unscaled["y_test"]
        preds = model.predict(X_test)
        predictions_test[name] = preds
        test_results[name] = compute_classification_metrics(y_test_curr, preds)

    return best_models, val_results, test_results, predictions_test


def run_classification_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes Experiment 1 across GBP, EUR, and JPY.
    Produces validation tables, test tables, and paper comparison table.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("=" * 75)
    logger.info("EXPERIMENT 1: SIGN PREDICTION (CLASSIFICATION)")
    logger.info("=" * 75)

    currencies = ["GBP", "EUR", "JPY"]
    all_val_records = []
    all_test_records = []
    selected_configs = {}

    for ccy in currencies:
        logger.info(f"\n--- Training & Tuning Models for {ccy} ---")
        ccy_data = load_split_data(ccy)
        best_models, val_res, test_res, _ = tune_and_train_models(ccy_data, random_state=42)

        selected_configs[ccy] = {m: best_models[m]["config"] for m in best_models}

        for model_name, metrics in val_res.items():
            record = {"Currency": ccy, "Model": model_name, "Config": best_models[model_name]["config"]}
            record.update(metrics)
            all_val_records.append(record)

        for model_name, metrics in test_res.items():
            record = {"Currency": ccy, "Model": model_name, "Config": best_models[model_name]["config"]}
            record.update(metrics)
            all_test_records.append(record)

            logger.info(
                f"[{ccy} TEST] {model_name:<14}: Acc={metrics['Acc']:.4f} | "
                f"DS={metrics['DS']:.4f} | CU={metrics['CU']:.4f} | CD={metrics['CD']:.4f} | "
                f"MAE={metrics['MAE']:.4f} | NMSE={metrics['NMSE']:.4f}"
            )

    val_df = pd.DataFrame(all_val_records)
    test_df = pd.DataFrame(all_test_records)

    # Save detailed validation and test tables
    val_path = os.path.join(TABLES_DIR, "classification_validation_results.csv")
    test_path = os.path.join(TABLES_DIR, "classification_test_results.csv")
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved validation table to {val_path}")
    logger.info(f"Saved test table to {test_path}")

    # Compute 3-Currency Averages (Directly matching Paper Table 1 format)
    metric_cols = ["Acc", "NMSE", "MAE", "DS", "CU", "CD", "Precision", "Recall", "F1"]
    avg_test = test_df.groupby("Model")[metric_cols].mean().round(4).reset_index()

    # Create Comparison Table against Paper Table 1
    comparison_records = []
    for model_name in ["Lasso", "Ridge", "Logistic", "SVMrbf", "Random Forest", "MLP"]:
        paper = PAPER_TABLE_1[model_name]
        reproduced = avg_test[avg_test["Model"] == model_name].iloc[0]

        comparison_records.append({
            "Model": model_name,
            "Paper_Acc": paper["Acc"],
            "Our_Acc": reproduced["Acc"],
            "Diff_Acc": round(reproduced["Acc"] - paper["Acc"], 4),
            "Paper_NMSE": paper["NMSE"],
            "Our_NMSE": reproduced["NMSE"],
            "Paper_MAE": paper["MAE"],
            "Our_MAE": reproduced["MAE"],
            "Paper_DS": paper["DS"],
            "Our_DS": reproduced["DS"],
            "Paper_CU": paper["CU"],
            "Our_CU": reproduced["CU"],
            "Paper_CD": paper["CD"],
            "Our_CD": reproduced["CD"],
        })

    comp_df = pd.DataFrame(comparison_records)
    comp_path = os.path.join(TABLES_DIR, "classification_paper_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved paper comparison table to {comp_path}")

    # Generate Comparison Visualizations
    generate_classification_plots(comp_df, test_df)

    return val_df, test_df, comp_df


def generate_classification_plots(comp_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Generate comparative visualization plots for classification models."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Bar Chart: Paper Reported Accuracy vs. Reproduced Test Accuracy
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(comp_df))
    width = 0.35

    ax.bar(x - width / 2, comp_df["Paper_Acc"], width, label="Paper (Stanford 2019)", color="#4c72b0", alpha=0.9)
    ax.bar(x + width / 2, comp_df["Our_Acc"], width, label="Our Reproduction (Test)", color="#55a868", alpha=0.9)

    ax.axhline(0.50, color="gray", linestyle="--", linewidth=1.0, label="Random Guess (50%)")
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Exchange Rate Sign Prediction: Paper Reported vs. Reproduced Accuracy (3-Currency Average)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_df["Model"], fontsize=10, fontweight="bold")
    ax.set_ylim(0.40, 0.60)
    ax.legend(loc="lower right", frameon=True)

    # Annotate bar values
    for i in x:
        ax.annotate(f"{comp_df['Paper_Acc'].iloc[i]:.2f}", (i - width / 2, comp_df["Paper_Acc"].iloc[i] + 0.005), ha="center", fontsize=9)
        ax.annotate(f"{comp_df['Our_Acc'].iloc[i]:.2f}", (i + width / 2, comp_df["Our_Acc"].iloc[i] + 0.005), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, "classification_accuracy_comparison.png")
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved accuracy comparison plot to {bar_path}")

    # 2. Per-Currency Test Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot_acc = test_df.pivot(index="Model", columns="Currency", values="Acc")[["GBP", "EUR", "JPY"]]
    pivot_acc.plot(kind="bar", ax=ax, colormap="viridis", width=0.7)
    ax.axhline(0.50, color="red", linestyle="--", linewidth=1.0, label="Random Guess (50%)")
    ax.set_ylabel("Test Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Test Accuracy by Currency Pair Across Reference Models", fontsize=12, fontweight="bold")
    ax.set_ylim(0.40, 0.60)
    ax.legend(title="Currency Pair", frameon=True)
    plt.xticks(rotation=0, fontweight="bold")
    plt.tight_layout()
    per_ccy_path = os.path.join(FIGURES_DIR, "classification_per_currency_accuracy.png")
    fig.savefig(per_ccy_path, dpi=200)
    plt.close(fig)
    logger.info(f"Saved per-currency accuracy plot to {per_ccy_path}")


if __name__ == "__main__":
    val_df, test_df, comp_df = run_classification_experiments()
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 CLASSIFICATION BENCHMARK SUMMARY (TABLE 1 REPRODUCTION)")
    print("=" * 80)
    print(comp_df.to_string(index=False))
