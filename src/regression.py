"""
src/regression.py

Experiment 2: Next-Day Return Regression for Foreign Currency Exchange Rate Forecast.
Replicates and benchmarks the machine learning regression models reported in Section 5.2,
Table 2, and Table 4 of the Stanford CS229 (Spring 2019) research paper:
1. Random Forest Regressor (1000 trees, max depth 100 per paper specification)
2. Multi-Layer Perceptron Regressor (MLP with 2 hidden layers per Section 4.3)
3. Support Vector Regression (SVR with RBF kernel)
4. Support Vector Regression (SVR with Polynomial kernel)
5. Lasso Regression (L1-regularized linear regression)
6. Ridge Regression (L2-regularized linear regression)

Protocol:
- Evaluates GBP, EUR, and JPY independently using chronological train, validation, and test splits.
- Scaled features used for linear models, SVR, and MLP; unscaled features for Random Forest.
- Hyperparameter tuning conducted strictly on train/validation sets (no test data leakage).
- Evaluates test set out-of-sample only after model selection.
- Computes R2 score, MSE, RMSE, and MAE across currencies.
- Contrasts measured results against Paper Table 2 and Table 4 reference values.
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.feature_engineering import ALL_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Paper Table 2 Reference Values (Cross-model benchmark reported in Stanford CS229 2019)
PAPER_TABLE_2 = {
    "Random Forest": 0.74,
    "MLP": "-inf",
    "SVM rbf": 0.00,
    "SVM poly": -0.02,
    "Lasso": 0.01,
    "Ridge": 0.01,
}

# Paper Table 4 Reference Values (Random Forest R^2 across currencies reported in paper)
PAPER_TABLE_4 = {
    "GBP": 0.70,
    "EUR": 0.78,
    "JPY": 0.74,
    "Average": 0.74,
}


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression evaluation metrics:
    - R^2 (coefficient of determination): 1 - sum((y - y_hat)^2) / sum((y - y_bar)^2)
    - MSE (Mean Squared Error)
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    var_y = float(np.var(y_true, ddof=0))
    if var_y < 1e-12:
        r2 = 0.0
    else:
        # Standard r2_score
        r2 = float(r2_score(y_true, y_pred))

    return {
        "R2": round(r2, 4),
        "MSE": round(mse, 8),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
    }


def load_split_data(
    currency_code: str, processed_dir: str = PROCESSED_DIR
) -> Dict[str, Dict[str, Any]]:
    """
    Loads train, validation, and test datasets for a given currency.
    Returns both unscaled (for RF) and scaled (for linear/SVR/MLP) representations.
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
            "y_train": train_raw["target_reg"].values,
            "X_val": val_raw[ALL_FEATURE_COLS],
            "y_val": val_raw["target_reg"].values,
            "X_test": test_raw[ALL_FEATURE_COLS],
            "y_test": test_raw["target_reg"].values,
            "dates_test": test_raw["DATE"].values,
            "rates_test": test_raw["rate"].values,
        },
        "scaled": {
            "X_train": train_sc[ALL_FEATURE_COLS],
            "y_train": train_sc["target_reg"].values,
            "X_val": val_sc[ALL_FEATURE_COLS],
            "y_val": val_sc["target_reg"].values,
            "X_test": test_sc[ALL_FEATURE_COLS],
            "y_test": test_sc["target_reg"].values,
            "dates_test": test_sc["DATE"].values,
            "rates_test": test_sc["rate"].values,
        },
    }


def tune_and_train_regression_models(
    currency_data: Dict[str, Dict[str, Any]], random_state: int = 42
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Trains candidate model configurations on training data, validates on validation data,
    selects best hyperparameter configurations based on validation R2 / MSE,
    and evaluates on out-of-sample test data.
    """
    unscaled = currency_data["unscaled"]
    scaled = currency_data["scaled"]

    y_val = scaled["y_val"]
    y_test = scaled["y_test"]

    best_models = {}
    val_results = {}
    test_results = {}
    predictions_test = {}

    # 1. Random Forest Regressor (Paper spec: 1000 trees, max depth 100, unscaled features)
    rf = RandomForestRegressor(
        n_estimators=1000,
        max_depth=100,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(unscaled["X_train"], unscaled["y_train"])
    best_models["Random Forest"] = {
        "model": rf,
        "config": "n_estimators=1000, max_depth=100",
        "scaled": False,
    }
    val_results["Random Forest"] = compute_regression_metrics(
        unscaled["y_val"], rf.predict(unscaled["X_val"])
    )

    # 2. MLP Regressor (2 hidden layers per Section 4.3 equation 1, scaled features)
    # Tune hidden sizes, alpha regularization, and activation strictly on validation set
    best_mlp, best_mlp_cfg, best_score_mlp = None, None, -np.inf
    mlp_candidates = [
        {"hidden": (64, 32), "alpha": 0.01, "act": "relu"},
        {"hidden": (64, 32), "alpha": 0.1, "act": "relu"},
        {"hidden": (64, 32), "alpha": 1.0, "act": "relu"},
        {"hidden": (100, 50), "alpha": 0.1, "act": "relu"},
        {"hidden": (100, 50), "alpha": 1.0, "act": "relu"},
        {"hidden": (32, 16), "alpha": 0.1, "act": "relu"},
        {"hidden": (32, 16), "alpha": 1.0, "act": "relu"},
    ]
    for cand in mlp_candidates:
        model = MLPRegressor(
            hidden_layer_sizes=cand["hidden"],
            alpha=cand["alpha"],
            activation=cand["act"],
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        model.fit(scaled["X_train"], scaled["y_train"])
        score = r2_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_mlp:
            best_score_mlp = score
            best_mlp_cfg = f"hidden={cand['hidden']}, alpha={cand['alpha']}"
            best_mlp = model

    best_models["MLP"] = {"model": best_mlp, "config": best_mlp_cfg, "scaled": True}
    val_results["MLP"] = compute_regression_metrics(y_val, best_mlp.predict(scaled["X_val"]))

    # 3. Support Vector Regression (SVR with RBF kernel, scaled features)
    best_svr_rbf, best_svr_rbf_cfg, best_score_rbf = None, None, -np.inf
    rbf_candidates = [
        {"C": 0.0001, "eps": 0.001},
        {"C": 0.0001, "eps": 0.005},
        {"C": 0.001, "eps": 0.001},
        {"C": 0.001, "eps": 0.005},
        {"C": 0.001, "eps": 0.01},
        {"C": 0.01, "eps": 0.005},
        {"C": 0.01, "eps": 0.01},
        {"C": 0.1, "eps": 0.005},
    ]
    for cand in rbf_candidates:
        model = SVR(kernel="rbf", C=cand["C"], epsilon=cand["eps"])
        model.fit(scaled["X_train"], scaled["y_train"])
        score = r2_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_rbf:
            best_score_rbf = score
            best_svr_rbf_cfg = f"kernel=rbf, C={cand['C']}, eps={cand['eps']}"
            best_svr_rbf = model

    best_models["SVM rbf"] = {"model": best_svr_rbf, "config": best_svr_rbf_cfg, "scaled": True}
    val_results["SVM rbf"] = compute_regression_metrics(y_val, best_svr_rbf.predict(scaled["X_val"]))

    # 4. Support Vector Regression (SVR with Polynomial kernel, scaled features)
    best_svr_poly, best_svr_poly_cfg, best_score_poly = None, None, -np.inf
    poly_candidates = [
        {"degree": 2, "C": 0.0001, "eps": 0.001},
        {"degree": 2, "C": 0.0001, "eps": 0.005},
        {"degree": 2, "C": 0.001, "eps": 0.001},
        {"degree": 2, "C": 0.001, "eps": 0.005},
        {"degree": 3, "C": 0.0001, "eps": 0.001},
        {"degree": 3, "C": 0.0001, "eps": 0.005},
    ]
    for cand in poly_candidates:
        model = SVR(kernel="poly", degree=cand["degree"], C=cand["C"], epsilon=cand["eps"])
        model.fit(scaled["X_train"], scaled["y_train"])
        score = r2_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_poly:
            best_score_poly = score
            best_svr_poly_cfg = f"kernel=poly, degree={cand['degree']}, C={cand['C']}, eps={cand['eps']}"
            best_svr_poly = model

    best_models["SVM poly"] = {"model": best_svr_poly, "config": best_svr_poly_cfg, "scaled": True}
    val_results["SVM poly"] = compute_regression_metrics(y_val, best_svr_poly.predict(scaled["X_val"]))

    # 5. Lasso Regression (L1 regularization, scaled features)
    best_lasso, best_lasso_cfg, best_score_lasso = None, None, -np.inf
    for alpha_cand in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
        model = Lasso(alpha=alpha_cand, random_state=random_state, max_iter=2000)
        model.fit(scaled["X_train"], scaled["y_train"])
        score = r2_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_lasso:
            best_score_lasso = score
            best_lasso_cfg = f"alpha={alpha_cand}"
            best_lasso = model

    best_models["Lasso"] = {"model": best_lasso, "config": best_lasso_cfg, "scaled": True}
    val_results["Lasso"] = compute_regression_metrics(y_val, best_lasso.predict(scaled["X_val"]))

    # 6. Ridge Regression (L2 regularization, scaled features)
    best_ridge, best_ridge_cfg, best_score_ridge = None, None, -np.inf
    for alpha_cand in [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        model = Ridge(alpha=alpha_cand, random_state=random_state)
        model.fit(scaled["X_train"], scaled["y_train"])
        score = r2_score(y_val, model.predict(scaled["X_val"]))
        if score > best_score_ridge:
            best_score_ridge = score
            best_ridge_cfg = f"alpha={alpha_cand}"
            best_ridge = model

    best_models["Ridge"] = {"model": best_ridge, "config": best_ridge_cfg, "scaled": True}
    val_results["Ridge"] = compute_regression_metrics(y_val, best_ridge.predict(scaled["X_val"]))

    # Out-of-sample Test Evaluation for all selected models
    for name, item in best_models.items():
        model = item["model"]
        X_test = scaled["X_test"] if item["scaled"] else unscaled["X_test"]
        y_test_curr = y_test if item["scaled"] else unscaled["y_test"]
        preds = model.predict(X_test)
        predictions_test[name] = preds
        test_results[name] = compute_regression_metrics(y_test_curr, preds)

    return best_models, val_results, test_results, predictions_test


def plot_regression_predictions(
    test_dates: np.ndarray,
    actual_returns: np.ndarray,
    predictions: Dict[str, np.ndarray],
    currency_code: str,
    output_path: str,
) -> None:
    """
    Generates actual vs. predicted return time-series visualizations (reproducing Figure 2 style).
    """
    plt.figure(figsize=(14, 6))
    plt.plot(actual_returns, label="Actual Log Return", color="black", alpha=0.6, linewidth=1.2)

    # Plot top models: Random Forest, Ridge, SVR rbf, MLP
    colors = {
        "Random Forest": "forestgreen",
        "Ridge": "crimson",
        "SVM rbf": "dodgerblue",
        "Lasso": "purple",
        "MLP": "orange",
    }
    for model_name, color in colors.items():
        if model_name in predictions:
            plt.plot(
                predictions[model_name],
                label=f"Predicted ({model_name})",
                color=color,
                alpha=0.75,
                linewidth=1.0,
            )

    # Format ticks: show date intervals
    step = max(1, len(test_dates) // 8)
    tick_indices = list(range(0, len(test_dates), step))
    tick_labels = [str(test_dates[i]) for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=30, ha="right", fontsize=9)

    plt.title(
        f"Experiment 2: Actual vs. Predicted Next-Day Return — {currency_code.upper()} (Test Set: 2017–2018)",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Next-Day Log Return (target_reg)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved return prediction plot to {output_path}")


def plot_r2_comparison(
    comparison_df: pd.DataFrame, output_path: str
) -> None:
    """
    Generates a grouped bar chart comparing our measured test R^2 against Paper Table 2 values.
    """
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting
    plot_data = []
    for _, row in comparison_df.iterrows():
        m = row["Model"]
        our_r2 = row["Our_R2"]
        # Handle paper R2 (which may be string '-inf' or float)
        paper_r2_raw = row["Paper_R2"]
        paper_r2 = -1.0 if paper_r2_raw == "-inf" else float(paper_r2_raw)

        plot_data.append({"Model": m, "Source": "Our Reproduced R² (3-ccy Avg)", "R2": our_r2})
        plot_data.append({"Model": m, "Source": "Paper Table 2 R² (Reported)", "R2": paper_r2})

    df_plot = pd.DataFrame(plot_data)

    sns.barplot(data=df_plot, x="Model", y="R2", hue="Source", palette=["#2b5c8f", "#d95f02"])
    plt.title("Experiment 2: Regression R² Comparison (Our Measured vs. Paper Table 2)", fontsize=12, fontweight="bold")
    plt.xlabel("Model", fontsize=10)
    plt.ylabel("R² Score (Clamped / Plotted)", fontsize=10)
    plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
    plt.ylim(-1.5, 1.0)
    plt.legend(loc="lower left", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved R2 comparison plot to {output_path}")


def run_regression_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes Experiment 2 across GBP, EUR, and JPY.
    Produces validation tables, test tables, and paper comparison tables.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("=" * 75)
    logger.info("EXPERIMENT 2: RETURN REGRESSION BENCHMARK")
    logger.info("=" * 75)

    currencies = ["GBP", "EUR", "JPY"]
    all_val_records = []
    all_test_records = []
    selected_configs = {}
    all_predictions = {}

    for ccy in currencies:
        logger.info(f"\n--- Training & Tuning Regression Models for {ccy} ---")
        ccy_data = load_split_data(ccy)
        best_models, val_res, test_res, preds = tune_and_train_regression_models(ccy_data, random_state=42)

        selected_configs[ccy] = {m: best_models[m]["config"] for m in best_models}
        all_predictions[ccy] = (ccy_data["unscaled"]["dates_test"], ccy_data["unscaled"]["y_test"], preds)

        for model_name, metrics in val_res.items():
            record = {"Currency": ccy, "Model": model_name, "Config": best_models[model_name]["config"]}
            record.update(metrics)
            all_val_records.append(record)

        for model_name, metrics in test_res.items():
            record = {"Currency": ccy, "Model": model_name, "Config": best_models[model_name]["config"]}
            record.update(metrics)
            all_test_records.append(record)

            logger.info(
                f"[{ccy} TEST] {model_name:<14}: R2={metrics['R2']:.4f} | "
                f"MSE={metrics['MSE']:.8f} | RMSE={metrics['RMSE']:.6f} | MAE={metrics['MAE']:.6f}"
            )

        # Plot actual vs predicted returns
        dates_te, actual_te, preds_dict = all_predictions[ccy]
        fig_path = os.path.join(FIGURES_DIR, f"regression_predictions_{ccy.lower()}.png")
        plot_regression_predictions(dates_te, actual_te, preds_dict, ccy, fig_path)

    val_df = pd.DataFrame(all_val_records)
    test_df = pd.DataFrame(all_test_records)

    # Save detailed validation and test tables
    val_path = os.path.join(TABLES_DIR, "regression_validation_results.csv")
    test_path = os.path.join(TABLES_DIR, "regression_test_results.csv")
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Saved validation table to {val_path}")
    logger.info(f"Saved test table to {test_path}")

    # Compute 3-Currency Averages
    metric_cols = ["R2", "MSE", "RMSE", "MAE"]
    avg_test = test_df.groupby("Model")[metric_cols].mean().round(6).reset_index()

    # Create Comparison Table against Paper Table 2
    comparison_records = []
    for model_name in ["Random Forest", "MLP", "SVM rbf", "SVM poly", "Lasso", "Ridge"]:
        paper_val = PAPER_TABLE_2[model_name]
        reproduced = avg_test[avg_test["Model"] == model_name].iloc[0]

        comparison_records.append({
            "Model": model_name,
            "Paper_R2": paper_val,
            "Our_R2": round(reproduced["R2"], 4),
            "Our_MSE": reproduced["MSE"],
            "Our_RMSE": reproduced["RMSE"],
            "Our_MAE": reproduced["MAE"],
        })

    comp_df = pd.DataFrame(comparison_records)
    comp_path = os.path.join(TABLES_DIR, "regression_paper_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved Paper Table 2 comparison table to {comp_path}")

    # Create Comparison Table against Paper Table 4 (RF across currencies)
    rf_per_ccy = test_df[test_df["Model"] == "Random Forest"].set_index("Currency")
    rf_avg_r2 = round(float(rf_per_ccy["R2"].mean()), 4)
    table_4_records = [
        {"Currency": "GBP", "Paper_RF_R2": PAPER_TABLE_4["GBP"], "Our_RF_R2": round(float(rf_per_ccy.loc["GBP", "R2"]), 4), "Our_RF_MSE": float(rf_per_ccy.loc["GBP", "MSE"])},
        {"Currency": "EUR", "Paper_RF_R2": PAPER_TABLE_4["EUR"], "Our_RF_R2": round(float(rf_per_ccy.loc["EUR", "R2"]), 4), "Our_RF_MSE": float(rf_per_ccy.loc["EUR", "MSE"])},
        {"Currency": "JPY", "Paper_RF_R2": PAPER_TABLE_4["JPY"], "Our_RF_R2": round(float(rf_per_ccy.loc["JPY", "R2"]), 4), "Our_RF_MSE": float(rf_per_ccy.loc["JPY", "MSE"])},
        {"Currency": "Average", "Paper_RF_R2": PAPER_TABLE_4["Average"], "Our_RF_R2": rf_avg_r2, "Our_RF_MSE": round(float(rf_per_ccy["MSE"].mean()), 8)},
    ]
    table4_df = pd.DataFrame(table_4_records)
    table4_path = os.path.join(TABLES_DIR, "regression_table4_rf_comparison.csv")
    table4_df.to_csv(table4_path, index=False)
    logger.info(f"Saved Paper Table 4 comparison table to {table4_path}")

    # Plot R2 comparison figure
    r2_fig_path = os.path.join(FIGURES_DIR, "regression_r2_comparison.png")
    plot_r2_comparison(comp_df, r2_fig_path)

    return val_df, test_df, comp_df


if __name__ == "__main__":
    run_regression_experiments()
