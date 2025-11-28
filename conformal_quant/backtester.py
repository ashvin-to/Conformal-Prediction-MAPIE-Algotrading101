"""Uncertainty-filtered quantitative trading strategy backtester using conformal intervals."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Quantitative strategy performance metrics."""
    total_trades: int
    filtered_trades: int
    win_rate: float
    cumulative_return: float
    sharpe_ratio: float
    max_drawdown: float
    coverage_rate: float


class ConformalTradingBacktester:
    """Backtest trading signals filtered by conformal prediction interval uncertainty bounds."""

    def __init__(self, risk_free_rate: float = 0.02) -> None:
        self.rf = risk_free_rate

    def run(
        self,
        y_true_returns: np.ndarray,
        y_pred_returns: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        min_expected_return: float = 0.001,
    ) -> tuple[BacktestResult, pd.DataFrame]:
        """Execute uncertainty-aware trade filtering.

        Rule: Only enter long position when lower_bound > 0 (statistically significant positive return).
              Only enter short position when upper_bound < 0.
        """
        y_true = np.asarray(y_true_returns).ravel()
        y_pred = np.asarray(y_pred_returns).ravel()
        lower = np.asarray(lower_bounds).ravel()
        upper = np.asarray(upper_bounds).ravel()
        n = len(y_true)

        signals = np.zeros(n)
        for i in range(n):
            if lower[i] > min_expected_return:
                signals[i] = 1.0  # High-confidence Long
            elif upper[i] < -min_expected_return:
                signals[i] = -1.0  # High-confidence Short

        strategy_returns = signals * y_true
        cum_returns = np.cumprod(1.0 + strategy_returns) - 1.0

        trades_executed = np.sum(signals != 0)
        filtered_out = np.sum(signals == 0)
        
        wins = np.sum(strategy_returns > 0)
        win_rate = float(wins / trades_executed) if trades_executed > 0 else 0.0

        daily_mean = float(np.mean(strategy_returns))
        daily_std = float(np.std(strategy_returns)) + 1e-9
        sharpe = float((daily_mean / daily_std) * np.sqrt(252))

        # Max drawdown
        cum_equity = np.cumprod(1.0 + strategy_returns)
        peak = np.maximum.accumulate(cum_equity)
        drawdowns = (cum_equity - peak) / peak
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        covered = (y_true >= lower) & (y_true <= upper)
        cov_rate = float(np.mean(covered))

        df_out = pd.DataFrame({
            "true_return": y_true,
            "pred_return": y_pred,
            "lower_bound": lower,
            "upper_bound": upper,
            "signal": signals,
            "strategy_return": strategy_returns,
            "cum_return": cum_returns,
        })

        res = BacktestResult(
            total_trades=int(trades_executed),
            filtered_trades=int(filtered_out),
            win_rate=round(win_rate, 4),
            cumulative_return=round(float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0, 4),
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 4),
            coverage_rate=round(cov_rate, 4),
        )

        return res, df_out
