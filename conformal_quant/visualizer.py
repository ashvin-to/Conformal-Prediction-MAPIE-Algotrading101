"""Stylized visualization suite for conformal prediction intervals and risk diagnostics."""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str = "Conformal Prediction Intervals",
    save_path: str | Path | None = None,
    max_samples: int = 60,
) -> None:
    """Plot sorted prediction intervals with ground truth and point forecasts."""
    n = min(len(y_true), max_samples)
    idx = np.argsort(y_true[:n])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(np.arange(n), y_true[:n][idx], "ko", markersize=4, label="Ground Truth ($y$)")
    ax.plot(np.arange(n), y_pred[:n][idx], color="#1f77b4", lw=1.8, label="Point Prediction ($\hat{y}$)")
    ax.fill_between(
        np.arange(n),
        lower[:n][idx],
        upper[:n][idx],
        color="#38bdf8",
        alpha=0.35,
        label="Conformal Prediction Band ($1 - \alpha$)",
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Sorted Sample Index")
    ax.set_ylabel("Target Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cumulative_equity(
    cum_returns: np.ndarray,
    benchmark_returns: np.ndarray | None = None,
    title: str = "Uncertainty-Filtered Trading Strategy Cumulative Return",
    save_path: str | Path | None = None,
) -> None:
    """Plot strategy equity curve against benchmark."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(cum_returns * 100, color="#10b981", lw=2, label="Conformal Filtered Strategy")
    if benchmark_returns is not None:
        ax.plot(benchmark_returns * 100, color="#6b7280", linestyle="--", label="Benchmark (Unfiltered)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Trading Periods (Days)")
    ax.set_ylabel("Cumulative Return (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
