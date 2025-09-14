"""Conformalized Classification producing prediction sets with marginal coverage guarantees."""

from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import MapieClassifier


class ConformalClassifier:
    """Wrapper around MAPIE Classifier providing set-valued predictions."""

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        method: Literal["score", "cumulated_score", "lac", "top_k"] = "score",
        cv: int | str = 5,
        confidence_level: float = 0.90,
    ) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.method = method
        self.cv = cv
        self.estimator = base_estimator or RandomForestClassifier(n_estimators=100, random_state=42)
        self.mapie = MapieClassifier(estimator=self.estimator, method=self.method, cv=self.cv)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ConformalClassifier:
        """Fit the classifier and calibrate nonconformity threshold."""
        self.mapie.fit(X, y)
        return self

    def predict_sets(
        self,
        X: np.ndarray | pd.DataFrame,
        alpha: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict standard class label and boolean prediction set matrix.

        Returns:
            (y_pred, y_pred_sets)
        """
        a = alpha if alpha is not None else self.alpha
        y_pred, y_pred_sets = self.mapie.predict(X, alpha=a)
        # y_pred_sets shape is (n_samples, n_classes, 1) or (n_samples, n_classes)
        if y_pred_sets.ndim == 3:
            y_pred_sets = y_pred_sets[:, :, 0]
        return y_pred, y_pred_sets
