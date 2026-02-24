# 📈 ConformalQuant — Uncertainty-Calibrated ML for Quantitative Finance & Trading

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/MAPIE-Conformal%20Prediction-blueviolet.svg)](https://mapie.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)

**ConformalQuant** is a distribution-free quantitative finance framework. By wrapping machine learning estimators with **Conformal Prediction**, it constructs calibrated prediction intervals and set-valued classifications with **exact finite-sample statistical coverage guarantees** ($1 - \alpha$), filtering out high-uncertainty trades.

---

## 🎯 Core Modules & Capabilities

1. **Conformalized Alpha Regressor (`conformal_quant.regression`)**:
   - Supports **CV+**, **Jackknife+**, and **Split Conformal** strategies.
   - Guaranteed coverage bounds without parametric Gaussian assumptions.

2. **Regime Classification Sets (`conformal_quant.classification`)**:
   - Outputs decision sets (e.g. `["Bull", "Volatile"]`) to prevent overconfident bets during regime transitions.

3. **Sequential Time Series Calibration (`conformal_quant.timeseries`)**:
   - Implements **EnbPI** (Ensemble Batch Prediction Intervals) adapting to non-stationary financial series.

4. **Uncertainty-Filtered Backtester (`conformal_quant.backtester`)**:
   - Executes trades only when prediction bounds are statistically distinguished from zero ($L_t > 0$ for Long, $U_t < 0$ for Short), dramatically reducing false-positive drawdowns.

5. **Diagnostic Metrics & Visualizer (`conformal_quant.metrics`, `visualizer`)**:
   - Evaluates empirical coverage (PICP), interval sharpness (MPIW), Winkler penalties, and generates publication-grade equity curves.

---

## 📂 Repository Structure

```text
ConformalQuant/
├── conformal_quant/
│   ├── __init__.py
│   ├── regression.py          <- Conformalized regressors (CV+, Jackknife+, Split)
│   ├── classification.py      <- Conformal classification sets with guaranteed coverage
│   ├── timeseries.py          <- Sequential time-series forecasting with EnbPI
│   ├── backtester.py          <- Uncertainty-aware strategy backtester
│   ├── visualizer.py          <- Publication-grade interval and equity plotting
│   └── metrics.py             <- PICP, MPIW (sharpness), Winkler score, set diagnostics
├── sales.csv                  <- Demand time series
├── conformal_prediction_mapie.ipynb  <- Tutorial notebook
├── run_demo.py                <- Comprehensive 3-experiment demo runner
├── requirements.txt
├── LICENSE.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Quantitative Demo
```bash
python run_demo.py
```

Sample output:
```text
=================================================================
  📈 ConformalQuant — Uncertainty-Calibrated Quantitative ML
=================================================================

[Experiment 1] Conformalized Return Regression (90% Confidence)
Target Coverage Rate      : 90.00%
Empirical Coverage (PICP) : 91.33%
Mean Interval Width (MPIW): 2.845% return spread
Winkler Score             : 0.03810

[Experiment 2] Conformalized Regime Classification (Prediction Sets)
Target Coverage Rate      : 90.00%
Empirical Coverage        : 90.56%
Decisive Predictions Rate : 82.22%

[Experiment 3] Uncertainty-Filtered Algorithmic Trading Strategy
High-Confidence Trades    : 94 (Filtered out 206 high-risk days)
Strategy Win Rate         : 68.09%
Annualized Sharpe Ratio   : 1.84
Max Strategy Drawdown     : -4.12%
=================================================================
```

---

## 📐 Mathematical Theory

### 1. Finite-Sample Coverage Guarantee
For exchangeable data and nonconformity scores $R_i$:
$$\mathbb{P}\left(Y_{n+1} \in \hat{C}_{1-\alpha}(X_{n+1})\right) \ge 1 - \alpha$$

### 2. Winkler Loss Function
$$\text{Score}_\alpha(L, U, y) = (U - L) + \frac{2}{\alpha}(L - y)\mathbb{I}(y < L) + \frac{2}{\alpha}(y - U)\mathbb{I}(y > U)$$

---

## 📜 License
MIT License.
