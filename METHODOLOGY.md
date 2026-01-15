# Methodology & Economic Rationale

This document outlines the quantitative framework, feature selection logic, and validation strategy used in this project. The goal is to predict 1-month forward stock returns using a blend of technical signals, fundamental risk factors, and momentum structure.

## 1. Feature Selection Philosophy

We adhere to a "Hybrid Alpha" approach, combining:
1.  **Technical Analysis:** To capture short-term supply/demand imbalances (Market Microstructure).
2.  **Fundamental Risk Factors:** To capture exposure to systematic economic drivers (Asset Pricing Theory).
3.  **Momentum Structure:** To capture behavioral persistence and trend acceleration (Behavioral Finance).

### A. Technical Indicators (Market Signals)
*Rationale: Prices often overextend due to liquidity crunches or psychological overreaction. We use normalized indicators to detect these mean-reversion opportunities.*

* **Bollinger Band Width:** Measures volatility. Sudden compression often precedes a breakout, while extreme expansion indicates overheating.
* **RSI (14-day):** A momentum oscillator used to identify overbought (>70) or oversold (<30) conditions.
* **Normalized MACD:** Measures trend direction. Unlike standard MACD, we normalize by price to ensure the feature is scale-invariant across different stock prices.

### B. Fundamental Risk Factors (Fama-French 3-Factor Model)
*Rationale: According to the Efficient Market Hypothesis (EMH), higher returns are often compensation for higher risk. We isolate "Alpha" (true skill) from "Beta" (risk exposure).*

We calculate **24-month Rolling Betas** against the Fama-French factors:
* **Market Beta ($\beta_{Mkt}$):** Sensitivity to the broad market. High beta = High volatility exposure.
* **Size Beta ($\beta_{SMB}$):** Exposure to the "Small minus Big" factor. Positive beta implies the stock behaves like a small-cap (often higher growth/risk).
* **Value Beta ($\beta_{HML}$):** Exposure to the "High minus Low" (Value vs. Growth) factor. Positive beta implies the stock acts like a Value stock; negative implies Growth.

### C. Temporal Context & Momentum
*Rationale: Stock trends exhibit autocorrelation. A single price point is insufficient; the "path" matters.*

* **Lagged Returns ($t-1$ to $t-6$):** Provides the model with recent history to detect mean reversion (e.g., "Did it crash last month?") or persistence.
* **Pure Momentum ($12m - 1m$):** Captures the long-term trend while removing the noise of the most recent month.
* **Trend Acceleration ($12m - 3m$):** Identifies if a trend is speeding up or slowing down.

---

## 2. Target Definition
The target variable is **Forward 1-Month Return**.
* We shift returns backward by 1 month so that at time $t$, the model predicts returns for $t+1$.
* We also compute 2, 3, 6, and 12-month targets for multi-horizon analysis, though the primary model focuses on the 1-month horizon.

---

## 3. Model Architecture

### Baseline: Ridge Regression (L2 Regularization)
* **Why:** Financial data is noisy. Ridge regression imposes a penalty on large coefficients, preventing the model from fitting to noise. It serves as a linear benchmark.
* **Metric:** Information Coefficient (IC) - The Spearman rank correlation between predicted and actual returns.

### Primary Model: Random Forest Regressor
* **Why:** Stock returns are non-linear. Interaction effects matter (e.g., "High RSI is bad, BUT if Momentum is also high, it might be a breakout").
* **Configuration:**
    * **Trees:** 500 (Robust averaging)
    * **Max Depth:** 3 (Strict regularization to prevent overfitting)
    * **Min Samples Leaf:** 20 (Ensures decisions are based on significant data clusters)

---

## 4. Validation Strategy: Purged K-Fold
Standard Cross-Validation fails in finance due to data leakage. We use **Purged K-Fold** to ensure rigorous testing:

1.  **No Shuffling:** Time series order is preserved.
2.  **Embargo (1%):** A gap is enforced after the test set to prevent "look-ahead bias" where information from the test set bleeds into the subsequent training fold.
3.  **Purging:** We remove training samples that overlap with the test set time window to ensure total separation.
