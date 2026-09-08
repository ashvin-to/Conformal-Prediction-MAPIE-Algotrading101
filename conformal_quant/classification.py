"""Conformalized Classification producing prediction sets with marginal coverage guarantees."""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import CrossConformalClassifier


class ConformalClassifier:
    """Wrapper around MAPIE Classifier providing set-valued predictions."""

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        cv: int = 5,
        confidence_level: float = 0.90,
    ) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.estimator = base_estimator or RandomForestClassifier(n_estimators=100, random_state=42)
        self.mapie = CrossConformalClassifier(estimator=self.estimator, cv=cv, confidence_level=confidence_level)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ConformalClassifier:
        """Fit the classifier and calibrate nonconformity threshold."""
        self.mapie.fit_conformalize(X, y)
        return self

    def predict_sets(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict standard class label and boolean prediction set matrix."""
        y_pred, y_pred_sets = self.mapie.predict_set(X)
        
        if y_pred_sets.ndim == 3:
            y_pred_sets = y_pred_sets[:, :, 0]
            
        return y_pred, y_pred_sets
