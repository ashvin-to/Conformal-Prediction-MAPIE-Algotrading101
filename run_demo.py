"""ConformalQuant: End-to-End Uncertainty-Calibrated Trading & ML Demo."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

from conformal_quant.regression import ConformalRegressor
from conformal_quant.classification import ConformalClassifier
from conformal_quant.backtester import ConformalTradingBacktester
from conformal_quant.visualizer import plot_prediction_intervals, plot_cumulative_equity
from conformal_quant.metrics import (
    prediction_interval_coverage,
    mean_prediction_interval_width,
    winkler_score,
    classification_set_metrics,
)


def run_regression_experiment() -> None:
    print("\n" + "=" * 60)
    print("  [Experiment 1] Conformalized Return Regression (90% Confidence)")
    print("=" * 60)
    
    np.random.seed(42)
    X, y = make_regression(n_samples=500, n_features=12, noise=4.0, random_state=42)
    # Scale to typical asset return magnitudes with heteroscedastic volatility
    y = (y / np.std(y)) * 0.02 + np.random.normal(0, np.abs(X[:, 0]) * 0.01)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = ConformalRegressor(confidence_level=0.90, method="plus", cv=5)
    reg.fit(X_train, y_train)
    y_pred, lower, upper = reg.predict_interval(X_test)

    y_pis = np.column_stack([lower, upper])
    cov = prediction_interval_coverage(y_test, y_pis)
    width = mean_prediction_interval_width(y_pis)
    wink = winkler_score(y_test, y_pis, alpha=0.10)

    print(f"Target Coverage Rate      : 90.00%")
    print(f"Empirical Coverage (PICP) : {cov * 100:.2f}%")
    print(f"Mean Interval Width (MPIW): {width * 100:.3f}% return spread")
    print(f"Winkler Score             : {wink:.5f}")

    plot_prediction_intervals(
        y_test, y_pred, lower, upper,
        title="ConformalQuant — Distribution-Free 90% Asset Return Intervals",
        save_path="outputs/figures/conformal_regression_intervals.png",
    )
    print("Saved interval chart to outputs/figures/conformal_regression_intervals.png")


def run_classification_experiment() -> None:
    print("\n" + "=" * 60)
    print("  [Experiment 2] Conformalized Regime Classification (Prediction Sets)")
    print("=" * 60)

    X, y = make_classification(
        n_samples=600,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = ConformalClassifier(confidence_level=0.90, method="score", cv=5)
    clf.fit(X_train, y_train)
    y_pred, pred_sets = clf.predict_sets(X_test)

    diag = classification_set_metrics(y_test, pred_sets)
    print(f"Target Coverage Rate      : 90.00%")
    print(f"Empirical Coverage        : {diag['empirical_coverage'] * 100:.2f}%")
    print(f"Mean Set Size             : {diag['mean_set_size']:.2f} regimes")
    print(f"Decisive Predictions Rate : {diag['singleton_rate'] * 100:.2f}%")
    print(f"Ambiguous Set Rate        : {(1 - diag['singleton_rate'] - diag['empty_rate']) * 100:.2f}%")


def run_backtest_experiment() -> None:
    print("\n" + "=" * 60)
    print("  [Experiment 3] Uncertainty-Filtered Algorithmic Trading Strategy")
    print("=" * 60)

    np.random.seed(42)
    n_days = 300
    true_alpha = np.random.normal(0.0005, 0.015, n_days)
    pred_alpha = true_alpha + np.random.normal(0, 0.010, n_days)
    uncertainty_width = np.abs(np.random.normal(0.015, 0.005, n_days))

    lower = pred_alpha - uncertainty_width
    upper = pred_alpha + uncertainty_width

    backtester = ConformalTradingBacktester()
    res, df = backtester.run(true_alpha, pred_alpha, lower, upper, min_expected_return=0.002)

    print(f"Total Periods Backtested  : {len(df)}")
    print(f"High-Confidence Trades    : {res.total_trades} (Filtered out {res.filtered_trades} high-risk days)")
    print(f"Strategy Win Rate         : {res.win_rate * 100:.2f}%")
    print(f"Cumulative Strategy Return: {res.cumulative_return * 100:+.2f}%")
    print(f"Annualized Sharpe Ratio   : {res.sharpe_ratio:.2f}")
    print(f"Max Strategy Drawdown     : {res.max_drawdown * 100:.2f}%")

    benchmark = np.cumprod(1.0 + np.sign(pred_alpha) * true_alpha) - 1.0
    plot_cumulative_equity(
        df["cum_return"].values,
        benchmark,
        title="ConformalQuant — Trading Strategy Equity Curve vs Naive Benchmark",
        save_path="outputs/figures/trading_strategy_backtest.png",
    )
    print("Saved backtest equity curve to outputs/figures/trading_strategy_backtest.png")


def main() -> None:
    print("=" * 65)
    print("  📈 ConformalQuant — Uncertainty-Calibrated Quantitative ML")
    print("=" * 65)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    run_regression_experiment()
    run_classification_experiment()
    run_backtest_experiment()
    print("\n" + "=" * 65)
    print("  ✅ All quantitative experiments finished successfully!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
