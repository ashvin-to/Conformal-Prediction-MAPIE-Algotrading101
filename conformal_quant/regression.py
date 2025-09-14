"""Conformalized Regression with finite-sample coverage guarantees."""

from __future__ import annotations
from typing import Any, Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mapie.regression import MapieRegressor


class ConformalRegressor:
    """Wrapper around MAPIE Regressor providing distribution-free prediction intervals."""

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        method: Literal["plus", "base", "minmax", "naive"] = "plus",
        cv: int | str = 5,
        confidence_level: float = 0.90,
    ) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.method = method
        self.cv = cv
        self.estimator = base_estimator or GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.mapie = MapieRegressor(estimator=self.estimator, method=self.method, cv=self.cv)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ConformalRegressor:
        """Fit the conformalized regressor and calibrate nonconformity scores."""
        self.mapie.fit(X, y)
        return self

    def predict_interval(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict point estimate, lower bound, and upper bound.

        Returns:
            (point_predictions, lower_bounds, upper_bounds)
        """
        a = alpha if alpha is not None else self.alpha
        y_pred, y_pis = self.mapie.predict(X, alpha=a)
        lower = y_pis[:, 0, 0] if y_pis.ndim == 3 else y_pis[:, 0]
        upper = y_pis[:, 1, 0] if y_pis.ndim == 3 else y_pis[:, 1]
        return y_pred, lower, upper
