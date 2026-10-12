"""
src/trading_simulation.py

Stage 6: Trading Strategy Simulation & Portfolio Backtesting.
Replicates and benchmarks the trading profit simulation reported in Section 5.3, Table 3,
and Figure 3 of the Stanford CS229 (Spring 2019) research paper:
"Foreign Currency Exchange Rate Forecast" (Jiahao Zhang, Jialu Sun, Jiequan Zhang).

Reported Paper Benchmarks (Table 3, starting with $100 cash):
- GBP: $104.40
- JPY: $104.44
- EUR: $110.50
- GBP + JPY: $100.33
- JPY + EUR: $109.46
- GBP + EUR: $104.88
- All Three Currencies (GBP + EUR + JPY): $101.30 (peaked at ~$112 in early 2018 per Figure 3)

Protocol & Rules:
1. Model: Uses Random Forest (1,000 trees, max depth 100) trained strictly on 2001-2015 historical data.
2. Signal: At time t, model outputs directional signal pred_clf in {0, 1} predicting next-day direction.
3. Position:
   - For a single currency:
     * If pred == 1 (predicting foreign currency appreciation / rate increase): hold foreign currency;
       realized daily return = P_{t+1} / P_t = target_ratio.
     * If pred == 0 (predicting rate drop / USD appreciation): hold USD cash;
       realized daily return = 1.0.
   - For multi-currency combinations:
     * Portfolio capital is divided equally among currencies signalling long (pred == 1).
     * If all signalling 0, portfolio stays 100% in USD cash (return = 1.0).
4. Quote Conventions:
   - GBP/USD: USD per GBP (direct quote). Return on GBP = P_{t+1} / P_t.
   - EUR/USD: USD per EUR (direct quote). Return on EUR = P_{t+1} / P_t.
   - USD/JPY: JPY per USD (indirect quote).
     * Reported along both conventions:
       (a) Quoted-rate long (holding USD against JPY, return = P_{t+1} / P_t).
       (b) Economic JPY long (holding JPY against USD, return = P_t / P_{t+1} = 1 / target_ratio).
5. Evaluation Period: Common test period (2017-03-07 to 2018-11-30, 438 trading days).
6. Execution without look-ahead: Decision at day t uses only features at day t, executed on subsequent price.
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

from sklearn.ensemble import RandomForestClassifier
from src.feature_engineering import ALL_FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
TABLES_DIR = os.path.join(PROJECT_ROOT, "results", "tables")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Paper Table 3 Reference Final Balances
PAPER_TABLE_3 = {
    "GBP": 104.40,
    "JPY": 104.44,
    "EUR": 110.50,
    "GBP + JPY": 100.33,
    "JPY + EUR": 109.46,
    "GBP + EUR": 104.88,
    "All Three (GBP + EUR + JPY)": 101.30,
}


def calculate_max_drawdown(balance_series: np.ndarray) -> float:
    """Computes maximum percentage drawdown from peak."""
    arr = np.asarray(balance_series, dtype=float)
    cummax = np.maximum.accumulate(arr)
    drawdowns = (arr - cummax) / cummax
    return float(np.min(drawdowns))


def train_rf_and_get_test_predictions(random_state: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Trains Random Forest classifier on training set and produces out-of-sample
    test set predictions and aligned price ratios.
    """
    preds_dict = {}
    currencies = ["GBP", "EUR", "JPY"]

    for ccy in currencies:
        ccy_lower = ccy.lower()
        train_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ccy_lower}_train.csv"))
        test_df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{ccy_lower}_test.csv"))

        rf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=100,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(train_df[ALL_FEATURE_COLS], train_df["target_clf"])

        test_out = test_df[["DATE", "rate", "target_ratio", "target_clf"]].copy()
        test_out["pred_clf"] = rf.predict(test_df[ALL_FEATURE_COLS])
        preds_dict[ccy] = test_out

    return preds_dict


def align_multi_currency_test_data(preds_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merges GBP, EUR, and JPY test observations on matching dates.
    Ensures perfect chronological row-by-row alignment across 438 test trading days.
    """
    gbp = preds_dict["GBP"][["DATE", "rate", "target_ratio", "pred_clf"]].rename(
        columns={"rate": "rate_gbp", "target_ratio": "ratio_gbp", "pred_clf": "pred_gbp"}
    )
    eur = preds_dict["EUR"][["DATE", "rate", "target_ratio", "pred_clf"]].rename(
        columns={"rate": "rate_eur", "target_ratio": "ratio_eur", "pred_clf": "pred_eur"}
    )
    jpy = preds_dict["JPY"][["DATE", "rate", "target_ratio", "pred_clf"]].rename(
        columns={"rate": "rate_jpy", "target_ratio": "ratio_jpy", "pred_clf": "pred_jpy"}
    )

    merged = gbp.merge(eur, on="DATE").merge(jpy, on="DATE")
    merged = merged.sort_values("DATE").reset_index(drop=True)
    return merged


def simulate_trading_strategy(
    df: pd.DataFrame,
    strategy_name: str,
    active_currencies: List[str],
    jpy_mode: str = "quote_long",  # 'quote_long' or 'econ_jpy'
    initial_cash: float = 100.0,
) -> Dict[str, Any]:
    """
    Simulates daily portfolio balance trajectory starting with initial_cash.

    Rules:
    - On day t, check active_currencies.
    - If a currency signals Long (pred == 1), it is eligible for investment.
    - Equal allocation among active signals:
        day_return = mean(daily returns of active currencies)
    - If no currencies signal Long on day t, entire portfolio stays in USD cash (day_return = 1.0).
    - Tracks daily balance, trades/position changes, peak, maximum drawdown, and buy-and-hold benchmarks.
    """
    balances = [initial_cash]
    daily_returns = []
    positions = []
    num_trades = 0
    prev_active = set()

    n = len(df)
    for i in range(n):
        active_returns = []
        current_active = set()

        for ccy in active_currencies:
            pred_col = f"pred_{ccy.lower()}"
            ratio_col = f"ratio_{ccy.lower()}"

            pred_val = df[pred_col].iloc[i]
            ratio_val = df[ratio_col].iloc[i]

            if ccy in ["GBP", "EUR"]:
                if pred_val == 1:
                    active_returns.append(ratio_val)
                    current_active.add(ccy)
            elif ccy == "JPY":
                if jpy_mode == "quote_long":
                    # Long USD/JPY quoted rate (long USD against JPY) when pred == 1
                    if pred_val == 1:
                        active_returns.append(ratio_val)
                        current_active.add(ccy)
                elif jpy_mode == "econ_jpy":
                    # Economic long JPY (appreciates when quoted rate decreases, pred == 0)
                    if pred_val == 0:
                        active_returns.append(1.0 / ratio_val)
                        current_active.add(ccy)

        # Count position changes / rebalancing trades
        if current_active != prev_active:
            num_trades += 1
            prev_active = current_active

        if len(active_returns) == 0:
            day_ret = 1.0  # 100% USD cash
            positions.append("Cash (USD)")
        else:
            day_ret = float(np.mean(active_returns))
            positions.append("+".join(sorted(current_active)))

        daily_returns.append(day_ret)
        balances.append(balances[-1] * day_ret)

    balance_trajectory = np.array(balances)
    final_balance = float(balance_trajectory[-1])
    total_return_pct = float(((final_balance - initial_cash) / initial_cash) * 100.0)
    max_dd = calculate_max_drawdown(balance_trajectory)
    peak_val = float(np.max(balance_trajectory))
    peak_idx = int(np.argmax(balance_trajectory))
    peak_date = str(df["DATE"].iloc[min(peak_idx, n - 1)])

    return {
        "Strategy": strategy_name,
        "Currencies": "+".join(active_currencies),
        "Initial_Cash": initial_cash,
        "Final_Balance": round(final_balance, 2),
        "Total_Return_Pct": round(total_return_pct, 2),
        "Max_Drawdown_Pct": round(max_dd * 100.0, 2),
        "Peak_Balance": round(peak_val, 2),
        "Peak_Date": peak_date,
        "Position_Changes": num_trades,
        "Trajectory": balance_trajectory,
        "Daily_Returns": np.array(daily_returns),
    }


def run_all_trading_simulations() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Executes all individual and multi-currency trading strategies reported in Paper Table 3.
    Produces results tables, paper comparison table, and Figure 3 balance trajectory plots.
    """
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    logger.info("=" * 75)
    logger.info("EXPERIMENT 3: TRADING PROFIT SIMULATION BACKTEST")
    logger.info("=" * 75)

    preds_dict = train_rf_and_get_test_predictions(random_state=42)
    aligned_df = align_multi_currency_test_data(preds_dict)
    logger.info(f"Aligned test data across {len(aligned_df)} trading days ({aligned_df['DATE'].iloc[0]} to {aligned_df['DATE'].iloc[-1]})")

    # Define all 7 paper strategies
    strategies_to_run = [
        ("GBP", ["GBP"]),
        ("JPY", ["JPY"]),
        ("EUR", ["EUR"]),
        ("GBP + JPY", ["GBP", "JPY"]),
        ("JPY + EUR", ["JPY", "EUR"]),
        ("GBP + EUR", ["GBP", "EUR"]),
        ("All Three (GBP + EUR + JPY)", ["GBP", "EUR", "JPY"]),
    ]

    sim_results = []
    trajectories = {}

    for strat_name, active_ccys in strategies_to_run:
        res = simulate_trading_strategy(aligned_df, strat_name, active_ccys, jpy_mode="quote_long")
        sim_results.append({
            "Strategy": res["Strategy"],
            "Initial_Balance": res["Initial_Cash"],
            "Final_Balance": res["Final_Balance"],
            "Total_Return_Pct": res["Total_Return_Pct"],
            "Max_Drawdown_Pct": res["Max_Drawdown_Pct"],
            "Peak_Balance": res["Peak_Balance"],
            "Peak_Date": res["Peak_Date"],
            "Position_Changes": res["Position_Changes"],
        })
        trajectories[strat_name] = res["Trajectory"]

        logger.info(
            f"[STRATEGY] {strat_name:<28}: Final=${res['Final_Balance']:6.2f} | "
            f"Return={res['Total_Return_Pct']:+5.2f}% | MaxDD={res['Max_Drawdown_Pct']:5.2f}% | "
            f"Peak=${res['Peak_Balance']:6.2f} ({res['Peak_Date']})"
        )

    # Buy-and-Hold Benchmarks for reference
    bh_gbp = 100.0 * (aligned_df["rate_gbp"].iloc[-1] / aligned_df["rate_gbp"].iloc[0])
    bh_eur = 100.0 * (aligned_df["rate_eur"].iloc[-1] / aligned_df["rate_eur"].iloc[0])
    bh_jpy = 100.0 * (aligned_df["rate_jpy"].iloc[-1] / aligned_df["rate_jpy"].iloc[0])

    sim_results.append({
        "Strategy": "Buy & Hold (GBP)",
        "Initial_Balance": 100.0,
        "Final_Balance": round(bh_gbp, 2),
        "Total_Return_Pct": round(bh_gbp - 100.0, 2),
        "Max_Drawdown_Pct": round(calculate_max_drawdown(100.0 * (aligned_df["rate_gbp"].values / aligned_df["rate_gbp"].iloc[0])) * 100.0, 2),
        "Peak_Balance": round(100.0 * (aligned_df["rate_gbp"].max() / aligned_df["rate_gbp"].iloc[0]), 2),
        "Peak_Date": str(aligned_df["DATE"].iloc[aligned_df["rate_gbp"].argmax()]),
        "Position_Changes": 1,
    })
    sim_results.append({
        "Strategy": "Buy & Hold (EUR)",
        "Initial_Balance": 100.0,
        "Final_Balance": round(bh_eur, 2),
        "Total_Return_Pct": round(bh_eur - 100.0, 2),
        "Max_Drawdown_Pct": round(calculate_max_drawdown(100.0 * (aligned_df["rate_eur"].values / aligned_df["rate_eur"].iloc[0])) * 100.0, 2),
        "Peak_Balance": round(100.0 * (aligned_df["rate_eur"].max() / aligned_df["rate_eur"].iloc[0]), 2),
        "Peak_Date": str(aligned_df["DATE"].iloc[aligned_df["rate_eur"].argmax()]),
        "Position_Changes": 1,
    })

    results_df = pd.DataFrame(sim_results)
    results_path = os.path.join(TABLES_DIR, "trading_simulation_results.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved trading simulation results to {results_path}")

    # Create Comparison Table against Paper Table 3
    comparison_records = []
    for strat_name, paper_val in PAPER_TABLE_3.items():
        matched_row = results_df[results_df["Strategy"] == strat_name].iloc[0]
        our_val = matched_row["Final_Balance"]
        diff = round(our_val - paper_val, 2)
        comparison_records.append({
            "Strategy": strat_name,
            "Paper_Final_Cash": paper_val,
            "Our_Final_Balance": our_val,
            "Diff_Cash": diff,
            "Our_Return_Pct": matched_row["Total_Return_Pct"],
            "Our_MaxDD_Pct": matched_row["Max_Drawdown_Pct"],
            "Our_Peak_Balance": matched_row["Peak_Balance"],
            "Our_Peak_Date": matched_row["Peak_Date"],
        })

    comp_df = pd.DataFrame(comparison_records)
    comp_path = os.path.join(TABLES_DIR, "trading_paper_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Saved Paper Table 3 comparison table to {comp_path}")

    # Generate Visualizations (Figure 3 Style)
    plot_trading_curves(aligned_df["DATE"].values, trajectories, comp_df)

    return results_df, comp_df, trajectories


def plot_trading_curves(
    dates: np.ndarray, trajectories: Dict[str, np.ndarray], comp_df: pd.DataFrame
) -> None:
    """
    Plots cumulative portfolio balance trajectories reproducing Figure 3 from paper.
    """
    # 1. Figure 3 Reproduction: Multi-Currency Strategy Portfolio Curve
    plt.figure(figsize=(13, 6))

    all_three_traj = trajectories["All Three (GBP + EUR + JPY)"]
    plt.plot(all_three_traj, label="All Three Currencies (GBP + EUR + JPY)", color="#1f77b4", linewidth=2.0)
    plt.plot(trajectories["GBP + EUR"], label="GBP + EUR", color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
    plt.plot(trajectories["EUR"], label="Single EUR", color="#ff7f0e", linestyle=":", linewidth=1.5, alpha=0.8)
    plt.axhline(100.0, color="gray", linestyle="-.", alpha=0.6, label="Initial Capital ($100)")

    step = max(1, len(dates) // 8)
    tick_indices = list(range(0, len(dates), step))
    tick_labels = [str(dates[i]) for i in tick_indices]
    plt.xticks(tick_indices, tick_labels, rotation=30, ha="right", fontsize=9)

    plt.title("Figure 3 Reproduction: Portfolio Cash Trajectory Over Time (Test Period: 2017–2018)", fontsize=12, fontweight="bold")
    plt.xlabel("Date", fontsize=10)
    plt.ylabel("Portfolio Balance (USD)", fontsize=10)
    plt.ylim(90, 125)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", framealpha=0.9, fontsize=9)
    plt.tight_layout()

    fig3_path = os.path.join(FIGURES_DIR, "trading_figure3_portfolio_cash.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved Figure 3 portfolio trajectory plot to {fig3_path}")

    # 2. Comprehensive Comparison of All 7 Strategies vs. Paper Table 3
    plt.figure(figsize=(11, 6))
    sns.barplot(data=comp_df, x="Strategy", y="Our_Final_Balance", color="#2b5c8f", alpha=0.85, label="Our Reproduced Balance")
    plt.scatter(
        range(len(comp_df)),
        comp_df["Paper_Final_Cash"],
        color="crimson",
        s=80,
        zorder=5,
        label="Paper Table 3 Reported Balance",
    )
    plt.axhline(100.0, color="black", linestyle="--", alpha=0.7, label="Initial $100 Cash")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    plt.ylabel("Final Balance ($)", fontsize=10)
    plt.title("Trading Simulation: Final Portfolio Balance vs. Paper Table 3", fontsize=12, fontweight="bold")
    plt.ylim(90, 120)
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()

    bar_fig_path = os.path.join(FIGURES_DIR, "trading_final_balance_comparison.png")
    plt.savefig(bar_fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved Table 3 comparison bar chart to {bar_fig_path}")


if __name__ == "__main__":
    run_all_trading_simulations()
