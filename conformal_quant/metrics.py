"""Metrics for evaluating conformal prediction intervals and classification sets."""

from __future__ import annotations
import numpy as np


def prediction_interval_coverage(y_true: np.ndarray, y_pis: np.ndarray) -> float:
    """Calculate empirical coverage probability (PICP).

    y_pis is expected to have shape (n_samples, 2, n_alphas) or (n_samples, 2).
    """
    y_true = np.asarray(y_true).ravel()
    if y_pis.ndim == 3:
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]
    elif y_pis.ndim == 2:
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]
    else:
        raise ValueError("Unexpected shape for y_pis")
    
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mean_prediction_interval_width(y_pis: np.ndarray) -> float:
    """Calculate mean prediction interval width (MPIW / sharpness)."""
    if y_pis.ndim == 3:
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]
    elif y_pis.ndim == 2:
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]
    else:
        raise ValueError("Unexpected shape for y_pis")
    
    return float(np.mean(np.maximum(0, upper - lower)))


def winkler_score(y_true: np.ndarray, y_pis: np.ndarray, alpha: float = 0.05) -> float:
    """Calculate Winkler score (joint measure of coverage and interval width).
    
    Lower Winkler score indicates better calibrated and sharper prediction intervals.
    """
    y_true = np.asarray(y_true).ravel()
    if y_pis.ndim == 3:
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]
    elif y_pis.ndim == 2:
        lower = y_pis[:, 0]
        upper = y_pis[:, 1]
    else:
        raise ValueError("Unexpected shape for y_pis")

    width = upper - lower
    under = y_true < lower
    over = y_true > upper

    score = width + (2.0 / alpha) * (lower - y_true) * under + (2.0 / alpha) * (y_true - upper) * over
    return float(np.mean(score))


def classification_set_metrics(y_true: np.ndarray, y_pred_sets: np.ndarray) -> dict[str, float]:
    """Calculate empirical coverage and average set size for conformal classification sets.
    
    y_pred_sets is a boolean matrix of shape (n_samples, n_classes).
    """
    y_true = np.asarray(y_true).ravel()
    n_samples = len(y_true)
    
    # Check if true class is in predicted set
    correct = [y_pred_sets[i, int(y_true[i])] for i in range(n_samples)]
    coverage = float(np.mean(correct))
    avg_size = float(np.mean(np.sum(y_pred_sets, axis=1)))
    singleton_rate = float(np.mean(np.sum(y_pred_sets, axis=1) == 1))
    empty_rate = float(np.mean(np.sum(y_pred_sets, axis=1) == 0))

    return {
        "empirical_coverage": round(coverage, 4),
        "mean_set_size": round(avg_size, 4),
        "singleton_rate": round(singleton_rate, 4),
        "empty_rate": round(empty_rate, 4),
    }
