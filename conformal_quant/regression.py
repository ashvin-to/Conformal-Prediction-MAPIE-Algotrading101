"""Conformalized Regression with finite-sample coverage guarantees (MAPIE 1.0+)."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import CrossConformalRegressor


class ConformalRegressor:
    """Wrapper around MAPIE Conformal Regressors providing distribution-free prediction intervals."""

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        cv: int = 5,
        confidence_level: float = 0.90,
    ) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.estimator = base_estimator or GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.mapie = CrossConformalRegressor(estimator=self.estimator, cv=cv, confidence_level=confidence_level)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ConformalRegressor:
        """Fit the base model and compute cross-conformal scores."""
        self.mapie.fit_conformalize(X, y)
        return self

    def predict_interval(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict point estimate, lower bound, and upper bound."""
        y_pred, y_pis = self.mapie.predict_interval(X)
        
        if y_pis.ndim == 3:
            lower = y_pis[:, 0, 0]
            upper = y_pis[:, 1, 0]
        else:
            lower = y_pis[:, 0]
            upper = y_pis[:, 1]
            
        return y_pred, lower, upper
