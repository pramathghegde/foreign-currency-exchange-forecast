# Foreign Currency Exchange Rate Forecast: Replication, Critical Audit, and XGBoost Extension of Stanford CS229 (2019)

**Authors:** Machine Learning Mini-Project Team  
**Institution:** Department of Computer Science and Engineering, PES University  
**Reference Paper:** *Foreign Currency Exchange Rate Forecast*, Jiahao Zhang, Jialu Sun, Jiequan Zhang (Stanford CS229, Spring 2019)  
**Date:** October 2026  
**Status:** Completed & Audited Final Project Report  

---

## Abstract

This report presents an independent replication, critical empirical audit, and algorithmic extension of the Stanford University CS229 (Spring 2019) project report titled *"Foreign Currency Exchange Rate Forecast"*. Using daily historical data (2001–2018) from the Federal Reserve Economic Data (FRED) and Yahoo Finance, we reconstruct the three major exchange rates (GBP/USD, EUR/USD, and USD/JPY) alongside 15 technical features and 8 macroeconomic indicators. We evaluate the paper's three original experiments under strict chronological out-of-sample partitioning: (1) directional sign prediction across six reference classification models, (2) next-day continuous log-return regression across six reference regression models, and (3) a quantitative currency switching trading simulation starting with \$100.00 capital. Furthermore, we implement **XGBoost** as a novel extension model not studied in the original work. 

Our empirical replication confirms that predicting daily directional currency changes remains bounded near random walk performance ($48.79\%$ to $50.08\%$ test accuracy vs. paper's $51.0\%$ to $55.0\%$). In regression, while regularized linear, SVM, and neural network models achieve expected financial return behavior ($R^2 \approx -0.05$ to $0.00$), we uncover that the paper's reported Random Forest $R^2 = 0.74$ fails out-of-sample ($R^2 = -7.22$ average, $-21.59$ on GBP) due to tree leaf extrapolation failure across post-Brexit non-stationary price regimes. The quantitative trading simulation successfully reproduces the macroeconomic peak and drawdown dynamics of Figure 3 (peaking at \$113.89 before ending at \$105.87 vs. paper's \$101.30). Finally, XGBoost demonstrates superior algorithmic resilience in regression ($R^2 = -0.0001$ on GBP, matching Lasso/Ridge) through gradient regularization, outperforming unpruned decision forests during distribution shifts.

---

## 1. Introduction & Problem Statement

Foreign currency exchange (Forex) rate forecasting is one of the most challenging domains in financial economics. According to the foundational *Meese-Rogoff puzzle* (1983), structural macroeconomic models systematically fail to outperform a naive random walk in out-of-sample short-term exchange rate forecasting. 

In Spring 2019, Stanford CS229 researchers Jiahao Zhang, Jialu Sun, and Jiequan Zhang investigated whether modern supervised machine learning could challenge this hypothesis by predicting daily movements for three major currency pairs:
1. **GBP/USD** (British Pound / US Dollar)
2. **EUR/USD** (Euro / US Dollar)
3. **USD/JPY** (US Dollar / Japanese Yen)

The authors evaluated six learning algorithms across sign prediction (classification), continuous return estimation (regression), and backtested a simulated trading strategy. However, the original code repository (`Jiequannnnnnnnnn/CS229-Exchangge-rate`) is permanently unavailable, leaving open questions regarding data leakage, quote conventions, hyperparameter selection, and the plausibility of their reported regression scores.

The objectives of this project are:
1. **Independently reproduce** the data pipeline, feature engineering, classification models, regression models, and trading simulation.
2. **Conduct an exhaustive methodological audit** to verify data leakage absence, mathematical consistency, and quote conventions.
3. **Extend the benchmark** by implementing and evaluating **XGBoost** (Extreme Gradient Boosting) for both classification and regression.
4. **Transparently document all discrepancies** between published claims and empirical measurements.

---

## 2. Original Paper & Related Work

### 2.1 Summary of Reference Paper
The reference paper investigated 18 years of historical data (January 2001 to November 2018) split chronologically:
- **Training Set:** 2001-01-02 to 2015-06-30 (~3,420 trading days)
- **Validation Set:** 2015-07-01 to 2017-03-02 (~422 trading days)
- **Test Set:** 2017-03-03 to 2018-11-30 (~438–440 trading days)

The paper explored three primary tasks:
- **Experiment 1 (Sign Prediction):** Evaluated Logistic Regression, Lasso, Ridge, SVM (RBF), Random Forest (1,000 trees, depth 100), and MLP (2 hidden layers). The paper reported 3-currency average accuracies ranging from $51.0\%$ to $55.0\%$ (Table 1).
- **Experiment 2 (Regression):** Evaluated Random Forest, MLP, SVM (RBF), SVM (Poly), Lasso, and Ridge. The paper reported $R^2 = 0.74$ for Random Forest, $-\infty$ for MLP, $0.00$ for SVM RBF, $-0.02$ for SVM Poly, and $0.01$ for Lasso and Ridge (Table 2).
- **Experiment 3 (Trading Strategy):** Simulated an active currency switching portfolio starting with \$100.00, reporting final balances of \$104.40 (GBP), \$104.44 (JPY), \$110.50 (EUR), and \$101.30 (All Three Currencies) (Table 3 and Figure 3).

### 2.2 Related Literature
- **Meese & Rogoff (1983):** Established that econometric models fail to outperform random walk benchmarks out-of-sample.
- **Engel & West (2005):** Demonstrated that asset prices reflect discounted expectations of future fundamentals, explaining why daily returns resemble white noise.
- **Breiman (2001) & Chen & Guestrin (2016):** Documented tree ensemble mechanisms, specifically noting that axis-aligned step functions cannot extrapolate outside the convex hull of training features.

---

## 3. Dataset Sources & Preprocessing

All raw datasets were ingested from primary financial repositories via automated caching scripts (`src/data_loader.py`):

| Series Code | Source | Ticker / Identifier | Frequency | Description | Clean Rows |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **GBP/USD** | FRED | `DEXUSUK` | Daily | USD per British Pound (direct quote) | 4,509 |
| **EUR/USD** | FRED | `DEXUSEU` | Daily | USD per Euro (direct quote) | 4,511 |
| **USD/JPY** | FRED | `DEXJPUS` | Daily | Japanese Yen per USD (indirect quote) | 4,509 |
| **Fed Funds** | FRED | `DFF` | Daily | Federal Funds Effective Rate | 6,574 |
| **CPI** | FRED | `CPIAUCNS` | Monthly | Consumer Price Index for All Urban Consumers | 216 |
| **Recession** | FRED | `USREC` | Monthly | NBER Recession Indicator (0 or 1) | 216 |
| **Federal Debt**| FRED | `GFDEBTN` | Quarterly | Total Public Debt (Millions USD) | 72 |
| **S&P 500** | Yahoo | `^GSPC` | Daily | S&P 500 Index Close | 4,527 |
| **S&P 100** | Yahoo | `^OEX` | Daily | S&P 100 Index Close | 4,527 |
| **NASDAQ** | Yahoo | `^IXIC` | Daily | NASDAQ Composite Close | 4,527 |
| **DJIA** | Yahoo | `^DJI` | Daily | Dow Jones Industrial Average Close | 4,527 |

### Missing Data & Alignment
Missing trading days (bank holidays) in FRED FX series were dropped prior to return computation. Low-frequency macroeconomic indicators (CPI, Debt, Recession) were forward-filled onto daily trading dates strictly after applying publication release lags.

---

## 4. Feature Engineering & Leakage Prevention

Following Section 3 of the paper, we constructed **23 total features** (`ALL_FEATURE_COLS`):

### 4.1 Technical Features (15 Features)
- **Log Returns:** Contemporaneous $r_t = \ln(P_t / P_{t-1})$ and previous $r_{t-1}$.
- **Return Moving Averages:** Rolling arithmetic means of log returns over 5, 10, 20, 60, and 120 days.
- **Log Price Polynomials:** Current log price $\ln(P_t)$, squared $(\ln(P_t))^2$, and cubed $(\ln(P_t))^3$.
- **Log Price Moving Averages:** Rolling means of $\ln(P)$ over 5, 10, 20, 60, and 220 days.

### 4.2 Financial & Macroeconomic Features (8 Features)
- **Equity Index Daily Returns:** Log returns of S&P 500, S&P 100, NASDAQ, and DJIA.
- **Monetary Policy:** Daily Fed Funds Effective Rate level (`DFF`).
- **Lagged Macro Growth:**
  - CPI monthly rate of change (`CPIAUCNS`) aligned with a **15-day publication lag** after month-end.
  - NBER recession indicator (`USREC`) aligned with a **30-day reporting lag**.
  - Quarterly federal debt growth (`GFDEBTN`) aligned with a **45-day reporting lag** after quarter-end.

### 4.3 Target Definitions & Leakage Verification
- **Continuous Return Target (`target_reg`):**
  $$r_{t+1} = \ln\left(\frac{P_{t+1}}{P_t}\right)$$
- **Directional Classification Target (`target_clf`):**
  $$y_{t+1} = \mathbb{I}(r_{t+1} > 0) \in \{0, 1\}$$
- **Realized Price Ratio (`target_ratio`):**
  $$\text{ratio}_{t+1} = \frac{P_{t+1}}{P_t} = \exp(r_{t+1})$$

All feature scalers (`StandardScaler`) were fitted exclusively on training data and applied out-of-sample to validation and test sets. Automated audit tests (`src/validate_features.py`) verified that no target columns or future price data exist in the feature matrices.

---

## 5. Experimental Setup & Chronological Splits

| Dataset Partition | Calendar Window | GBP Rows | EUR Rows | JPY Rows |
| :--- | :---: | :---: | :---: | :---: |
| **Train Split** | 2001-01-02 to 2015-06-30 | 3,421 | 3,419 | 3,421 |
| **Validation Split** | 2015-07-01 to 2017-03-02 | 422 | 422 | 422 |
| **Test Split** | 2017-03-03 / 07 to 2018-11-30 | 438 | 440 | 438 |

All models were evaluated on the exact test period specified in the Stanford paper. Hyperparameter selection was conducted strictly on the validation set.

---

## 6. Experiment 1: Sign Prediction (Classification)

### 6.1 Reference Model Implementations
1. **Logistic Regression:** Regularization parameter $C \in [0.01, 10.0]$ tuned via validation accuracy.
2. **Lasso Classification:** $L1$-penalized logistic regression solved via `liblinear`.
3. **Ridge Classification:** $L2$-penalized linear classification via `RidgeClassifier`.
4. **Support Vector Machine (SVM RBF):** $C$ and $\gamma$ tuned via grid search on validation set.
5. **Random Forest:** 1,000 estimators, max depth 100, unscaled features (exact paper spec).
6. **Multi-Layer Perceptron (MLP):** 2 hidden layers ($(64, 32)$, $(100, 50)$, $(32, 16)$) with early stopping.

### 6.2 Comparison Against Paper Table 1 (3-Currency Test Set Averages)

| Model | Paper Acc | Our Acc | Diff ($\Delta$) | Paper NMSE | Our NMSE | Paper MAE | Our MAE | Paper DS | Our DS | Paper CU | Our CU | Paper CD | Our CD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 0.53 | **0.5008** | -0.0292 | 2.03 | **1.9972** | 1.02 | **0.9984** | 0.49 | **0.5008** | 0.44 | **0.7083** | 0.49 | **0.2903** |
| **MLP** | 0.52 | **0.4939** | -0.0261 | 1.92 | **2.0248** | 0.96 | **1.0122** | 0.52 | **0.4939** | 0.54 | **0.5643** | 0.52 | **0.4242** |
| **Logistic** | 0.51 | **0.4932** | -0.0168 | 1.97 | **2.0277** | 0.98 | **1.0137** | 0.51 | **0.4932** | 0.45 | **0.7466** | 0.57 | **0.2373** |
| **Ridge** | 0.55 | **0.4909** | -0.0591 | 1.81 | **2.0368** | 0.91 | **1.0182** | 0.55 | **0.4909** | 0.58 | **0.7404** | 0.51 | **0.2389** |
| **SVMrbf** | 0.51 | **0.4901** | -0.0199 | 1.95 | **2.0400** | 0.97 | **1.0198** | 0.51 | **0.4901** | 0.57 | **0.4973** | 0.52 | **0.4867** |
| **Lasso** | 0.52 | **0.4879** | -0.0321 | 1.90 | **2.0490** | 0.95 | **1.0243** | 0.52 | **0.4879** | 0.47 | **0.4272** | 0.58 | **0.5558** |

*Artifacts: See `results/tables/classification_paper_comparison.csv` and `results/figures/classification_accuracy_comparison.png`.*

### 6.3 Findings
Out-of-sample directional accuracy across all models clusters tightly between $48.8\%$ and $50.1\%$. The paper's reported modest edge ($51\%$ to $55\%$) is largely within sample variance limits of a 438-day test sample ($\sigma \approx 2.4\%$). Directional symmetry (DS) matches accuracy, and normalized error metrics (NMSE $\approx 2.0$, MAE $\approx 1.0$) conform to theoretical bounds on $\{-1, +1\}$ directional predictions.

---

## 7. Experiment 2: Numerical Return Regression

### 7.1 Reference Model Implementations
1. **Random Forest Regressor:** 1,000 estimators, max depth 100, unscaled features.
2. **MLP Regressor:** 2 hidden layers, validation-tuned architecture and $L2$ penalty.
3. **SVR (RBF Kernel):** $C \in [10^{-4}, 10^{-1}]$, $\epsilon \in [10^{-3}, 10^{-2}]$ tuned on validation.
4. **SVR (Polynomial Kernel):** Degrees 2 & 3, $C \in [10^{-4}, 10^{-2}]$, $\epsilon \in [10^{-3}, 10^{-2}]$.
5. **Lasso Regressor:** $\alpha \in [10^{-5}, 10^{-2}]$ tuned on validation.
6. **Ridge Regressor:** $\alpha \in [0.1, 5000.0]$ tuned on validation.

### 7.2 Comparison Against Paper Table 2 (3-Currency Test Set $R^2$)

| Model | Paper Table 2 $R^2$ | Our Measured $R^2$ (3-ccy Avg) | Our MSE | Our RMSE | Our MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | `0.74` | **-7.2231** | $0.000196$ | $0.010739$ | $0.009851$ |
| **MLP** | `-inf` | **-0.0057** | $0.000021$ | $0.004528$ | $0.003522$ |
| **SVM rbf** | `0.00` | **-0.0498** | $0.000021$ | $0.004621$ | $0.003592$ |
| **SVM poly** | `-0.02` | **-0.0061** | $0.000021$ | $0.004529$ | $0.003514$ |
| **Lasso** | `0.01` | **-0.0017** | $0.000020$ | $0.004518$ | $0.003507$ |
| **Ridge** | `0.01` | **-0.0107** | $0.000021$ | $0.004538$ | $0.003526$ |

### 7.3 Comparison Against Paper Table 4 (Random Forest Across Currencies)

| Currency | Paper Table 4 RF $R^2$ | Our Measured RF $R^2$ | Our Measured RF MSE |
| :--- | :---: | :---: | :---: |
| **GBP** | 0.70 | **-21.5947** | $0.000550$ |
| **EUR** | 0.78 | **-0.0505** | $0.000019$ |
| **JPY** | 0.74 | **-0.0242** | $0.000019$ |
| **Average** | 0.74 | **-7.2231** | $0.000196$ |

*Artifacts: See `results/tables/regression_paper_comparison.csv`, `results/tables/regression_table4_rf_comparison.csv`, and `results/figures/regression_r2_comparison.png`.*

---

## 8. Investigation of the Random Forest $R^2$ Discrepancy

A central finding of our audit is the stark discrepancy between the paper's reported Random Forest $R^2 = 0.74$ and our measured out-of-sample $R^2 = -7.22$ (with GBP reaching $-21.59$). We conducted a deep structural investigation to determine the exact mathematical cause.

### 8.1 In-Sample Overfitting vs. Out-of-Sample Evaluation
When evaluating the identical Random Forest model (1,000 trees, depth 100) on the **training set**, the model achieves:
$$\text{Train } R^2_{\text{GBP}} = 0.8661, \quad \text{Train } R^2_{\text{EUR}} = 0.8614, \quad \text{Train } R^2_{\text{JPY}} = 0.8634$$
Because the paper did not provide code, it is plausible that the authors' reported $0.70$–$0.78$ represented in-sample fit or non-chronological cross-validation.

### 8.2 Decision Tree Leaf Extrapolation Failure Across Regime Shifts
For EUR and JPY, the test set $R^2$ values are $-0.0505$ and $-0.0242$, consistent with standard financial return noise. However, GBP suffered an catastrophic collapse to $-21.59$.
1. **The Brexit Regime Shift:** Following the June 2016 Brexit referendum, GBP/USD traded at 30-year historic lows ($1.20$–$1.35$), corresponding to log prices of $\ln P \in [0.19, 0.36]$. In the training period (2001–2015), GBP had never traded at these levels ($\min(\ln P_{\text{train}}) = 0.3117$).
2. **Extreme Leaf Assignment:** $100\%$ of GBP test observations had 220-day moving average prices below the training set minimum. In the decision trees, these points routed to the outermost leaves associated with the 2008–2009 financial crisis rebound shock, where the training target was $+4.27\%$ daily return.
3. **Systematic Bias:** The Random Forest predicted an average daily return of **$+2.27\%$** for GBP over the entire 2017–2018 test set (actual mean: $+0.01\%$). Because $(y - \hat{y})^2 \gg \text{Var}(y)$, the resulting $R^2$ was $-21.59$.
4. **Linear & Regularized Models:** In contrast, Lasso, Ridge, MLP, and SVM did not partition test data into extreme step functions, achieving $R^2 \approx -0.01$ to $0.00$ with tiny MSE values ($2.4 \times 10^{-5}$).

---

## 9. Trading Simulation Methodology & Results

### 9.1 Strategy Rules & Execution Horizon
Following Section 5.3 of the paper, we simulated daily currency switching over the 438-day test period (2017-03-07 to 2018-11-30) starting with **\$100.00 initial capital**:
- **Signal:** At close $t$, the Random Forest classifier predicts direction $\hat{y}_{t+1} \in \{0, 1\}$.
- **Single Currency:** If $\hat{y} = 1$, hold foreign currency (realized return $P_{t+1}/P_t$); if $\hat{y} = 0$, hold USD cash (realized return $1.00$).
- **Multi-Currency Combinations:** Portfolio capital is allocated equally among currencies signalling long. If all signal down, capital remains 100% in USD cash.
- **Quote Conventions:** Direct quotes for GBP/USD and EUR/USD. For USD/JPY, quoted long positions evaluate USD appreciation against JPY.

### 9.2 Comparison Against Paper Table 3

| Strategy | Paper Table 3 Final Cash | Our Measured Final Balance | Difference ($\Delta$) | Total Return | Max Drawdown | Peak Balance | Peak Date |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GBP** | **$104.40** | **$106.46** | +$2.06 | +6.46% | -10.86% | $118.54 | 2018-02-01 |
| **JPY** | **$104.44** | **$96.84** | -$7.60 | -3.16% | -7.96% | $100.90 | 2017-03-10 |
| **EUR** | **$110.50** | **$106.29** | -$4.21 | +6.29% | -8.30% | $114.46 | 2018-02-15 |
| **GBP + JPY** | **$100.33** | **$105.18** | +$4.85 | +5.18% | -10.39% | $112.54 | 2018-02-01 |
| **JPY + EUR** | **$109.46** | **$103.24** | -$6.22 | +3.24% | -7.41% | $106.82 | 2018-02-01 |
| **GBP + EUR** | **$104.88** | **$108.86** | +$3.98 | +8.86% | -10.54% | $120.99 | 2018-04-18 |
| **All Three (GBP+EUR+JPY)** | **$101.30** | **$105.87** | +$4.57 | +5.87% | -8.86% | **$113.89** | 2018-02-01 |
| *Buy & Hold (GBP)* | *N/A* | *$104.61* | — | *+4.61%* | *-11.49%* | *$117.39* | *2018-04-16* |
| *Buy & Hold (EUR)* | *N/A* | *$107.00* | — | *+7.00%* | *-9.67%* | *$118.01* | *2018-01-25* |

*Artifacts: See `results/tables/trading_paper_comparison.csv` and `results/figures/trading_figure3_portfolio_cash.png`.*

### 9.3 Reproduction of Figure 3 Trajectory Dynamics
Our reproduced 3-currency portfolio trajectory exhibited the exact qualitative behavior illustrated in Figure 3 of the paper:
- Rapid capital appreciation during year one, peaking at **\$113.89** in February 2018 (matching the paper's reported ~\$112 peak).
- A substantial macroeconomic regime change beginning in spring 2018, eroding portfolio gains to finish at **\$105.87** (paper: \$101.30).

---

## 10. XGBoost Extension & Comparison

We implemented **XGBoost** (v3.4.1) as an algorithmic extension to test whether gradient boosting with explicit tree regularization and shrinkage outperforms the reference models.

### 10.1 XGBoost Classification Performance
- **Tuned Hyperparameters:** Tree depth 3–5, learning rate $0.03$–$0.10$, subsample $0.7$, colsample $0.8$, $L1$ regularization $\alpha = 0.1$.
- **Test Set Accuracy:** **$49.24\%$** (GBP: $50.00\%$, EUR: $47.73\%$, JPY: $50.00\%$).
- **Comparison:** Matches the reference models ($48.79\%$ to $50.08\%$), confirming that gradient boosted trees cannot overcome the random walk limit without alternative non-public information.

### 10.2 XGBoost Regression Performance & Regularization Advantage
- **Tuned Hyperparameters:** Shallow trees (depth 2–5), small shrinkage ($\eta = 0.01$–$0.05$), and heavy $L1$ leaf regularization ($\alpha = 0.01$–$1.0$).
- **Test Set $R^2$:** **$-0.0109$** average (GBP: **$-0.0001$**, EUR: **$-0.0246$**, JPY: **$-0.0081$**).
- **Critical Finding:** While unpruned Random Forest diverged on GBP ($R^2 = -21.59$), XGBoost's additive tree shrinkage and leaf regularization prevented extreme extrapolation, keeping GBP test $R^2$ at **$-0.0001$** (MSE $2.433 \times 10^{-5}$), matching Lasso and Ridge.

### 10.3 Feature Importance Analysis
Feature importance scores ([`results/figures/xgboost_feature_importance.png`](file:///d:/Engineering/PESU/SEM%205/Machine%20Learning/ML%20Mini%20Project/results/figures/xgboost_feature_importance.png)) demonstrate that XGBoost primarily leverages:
1. Short-term momentum: `return_ma_10`, `return_ma_5`, `return_t_1`
2. Interest rate policy: `fed_funds_rate`
3. Equity index spillovers: `nasdaq_return`, `sp500_return`
Non-stationary log price polynomials (`log_price`, `log_price_cube`) received substantially reduced gain weights, explaining why XGBoost remained stable across regime shifts.

---

## 11. Cross-Model Synthesis Benchmark

| Model | Model Family | Directional Accuracy | Regression $R^2$ (3-ccy Avg) | Regression MSE | Robust to Regime Shifts? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Lasso** | Linear ($L1$) | $48.79\%$ | **-0.0017** | $2.0 \times 10^{-5}$ | Yes |
| **MLP** | Neural Network | $49.39\%$ | **-0.0057** | $2.1 \times 10^{-5}$ | Yes (with early stopping) |
| **XGBoost** | Gradient Boosted Trees | $49.24\%$ | **-0.0109** | $2.1 \times 10^{-5}$ | Yes (with shrinkage & $L1$) |
| **Ridge** | Linear ($L2$) | $49.09\%$ | **-0.0107** | $2.1 \times 10^{-5}$ | Yes |
| **SVM (Poly)**| Kernel Machine | N/A | **-0.0061** | $2.1 \times 10^{-5}$ | Yes |
| **SVM (RBF)** | Kernel Machine | $49.01\%$ | **-0.0498** | $2.1 \times 10^{-5}$ | Yes |
| **Logistic** | Linear | $49.32\%$ | N/A | N/A | Yes |
| **Random Forest**| Bagged Trees | **50.08%** | **-7.2231** | $19.6 \times 10^{-5}$ | **No (fails on price shifts)** |

*Artifacts: See `results/tables/xgboost_paper_comparison.csv`, `results/figures/xgboost_vs_reference_classification.png`, and `results/figures/xgboost_vs_reference_regression.png`.*

---

## 12. Limitations & Threats to Validity

1. **Transaction Costs & Bid-Ask Spreads:** The simulation follows the paper by omitting transaction costs. In real-world FX markets, frequent daily switching (150–250 position changes) would incur spreads (1–2 pips) and slippage that could erode net profits.
2. **Short Out-of-Sample Test Horizon:** The test period spans 438 trading days (March 2017 to November 2018). While identical to the paper, this is too brief to confirm statistical significance across diverse macro cycles.
3. **Unresolved Paper Implementation Details:** Due to the unavailability of the authors' original repository, exact random seeds, software package versions, and hyperparameter grids could not be verified directly.
4. **Non-Stationarity of Level Features:** Including raw log prices and price moving averages creates severe distribution shifts for tree-based models when assets establish new historical highs or lows.

---

## 13. Conclusion & Future Work

This project successfully reproduced and critically evaluated the Stanford CS229 (2019) exchange rate forecasting benchmark:
- **Directional Prediction:** Machine learning models achieve $\sim 49\%$–$50\%$ accuracy, verifying that daily exchange rate directions closely follow a martingale difference sequence.
- **Regression Discrepancy:** We demonstrated that the paper's reported Random Forest $R^2 = 0.74$ is not reproducible under chronological out-of-sample evaluation ($R^2 = -7.22$), resulting from tree extrapolation failure over non-stationary price features.
- **Trading Simulation:** The currency switching strategy successfully mirrored the empirical rise, peak (\$113.89), and regime-shift decay of Figure 3, achieving a final balance of \$105.87 (paper: \$101.30).
- **XGBoost Extension:** Gradient boosted trees demonstrated superior stability over Random Forest in regression ($R^2 = -0.0001$ on GBP) by leveraging additive regularization and feature shrinkage.

### Future Work
Future research should explore:
1. Stationary feature engineering (replacing log price levels with fractional differentiation or volatility-adjusted ratios).
2. Deep sequential models (LSTMs, GRUs, and Temporal Transformers) trained with multi-task loss.
3. Realistic transaction cost modeling and Kelly-criterion position sizing.

---

## 14. References

1. Zhang, J., Sun, J., & Zhang, J. (2019). *Foreign Currency Exchange Rate Forecast*. Stanford University CS229 Project Report. [http://cs229.stanford.edu/proj2019spr/report/48.pdf](http://cs229.stanford.edu/proj2019spr/report/48.pdf)
2. Meese, R. A., & Rogoff, K. (1983). *Empirical exchange rate models of the seventies: Do they fit out of sample?* Journal of International Economics, 14(1-2), 3-24.
3. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
5. Engel, C., & West, K. D. (2005). *Exchange Rates and Fundamentals*. Journal of Political Economy, 113(3), 485-517.

---

## 15. Reproducibility Instructions

### Environment Setup
```powershell
# Create project-local virtual environment
python -m venv .venv --system-site-packages

# Install required dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt
```

### End-to-End Pipeline Execution
```powershell
# 1. Download & cache historical raw datasets
.\.venv\Scripts\python.exe src/data_loader.py

# 2. Run feature engineering and dataset splitting
.\.venv\Scripts\python.exe src/feature_engineering.py

# 3. Execute Experiment 1: Sign Prediction (Classification)
.\.venv\Scripts\python.exe src/classification.py

# 4. Execute Experiment 2: Numerical Return Regression
.\.venv\Scripts\python.exe src/regression.py

# 5. Execute Experiment 3: Trading Strategy Simulation
.\.venv\Scripts\python.exe src/trading_simulation.py

# 6. Execute Stage 7: XGBoost Extension
.\.venv\Scripts\python.exe src/xgboost_models.py
```

### Complete Test & Validation Suite
```powershell
# Run complete project verification suite (all 6 modules):
.\.venv\Scripts\python.exe -c "
import subprocess, sys
scripts = [
    'src/validate_data.py',
    'src/validate_features.py',
    'src/validate_classification.py',
    'src/validate_regression.py',
    'src/validate_trading_simulation.py',
    'src/validate_xgboost.py',
]
for s in scripts:
    res = subprocess.run([sys.executable, s])
    assert res.returncode == 0, f'Failed on {s}'
print('ALL 6 VALIDATION SUITES PASSED (0 ERRORS)')
"
```
