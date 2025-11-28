"""Time-Series Conformal Forecaster with MAPIE EnbPI / Sequential Calibration."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from mapie.time_series_regressors import MapieTimeSeriesRegressor


class ConformalTimeSeriesForecaster:
    """Sequential time-series conformal prediction interval forecaster."""

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        method: str = "enbpi",
        cv: int = 5,
        confidence_level: float = 0.90,
    ) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.method = method
        self.cv = cv
        self.estimator = base_estimator or GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.mapie = MapieTimeSeriesRegressor(estimator=self.estimator, method=self.method, cv=self.cv)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ConformalTimeSeriesForecaster:
        """Fit sequential conformal regressor on ordered time series."""
        self.mapie.fit(X, y)
        return self

    def predict_interval(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float | None = None,
        ensemble: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict time-series point forecasts and calibrated conformal prediction intervals."""
        a = alpha if alpha is not None else self.alpha
        y_pred, y_pis = self.mapie.predict(X, alpha=a, ensemble=ensemble)
        lower = y_pis[:, 0, 0] if y_pis.ndim == 3 else y_pis[:, 0]
        upper = y_pis[:, 1, 0] if y_pis.ndim == 3 else y_pis[:, 1]
        return y_pred, lower, upper
