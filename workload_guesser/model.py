"""ML model for predicting course workload (low / medium / high).

The :class:`WorkloadPredictor` wraps a scikit-learn ``Pipeline`` that combines
the feature extraction steps defined in :mod:`workload_guesser.features` with a
``RandomForestClassifier``.  Instances can be serialised to disk with
:meth:`WorkloadPredictor.save` and restored with :meth:`WorkloadPredictor.load`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from workload_guesser.data import load_courses
from workload_guesser.features import build_feature_pipeline

#: Ordered list of workload category labels used throughout the package.
WORKLOAD_LABELS = ["low", "medium", "high"]


class WorkloadPredictor:
    # train and use a course-workload prediction model.


    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None

    def train(
        self,
        data_path: Optional[str | Path] = None,
        *,
        cv: bool = False,
    ) -> "WorkloadPredictor":
        """Train the model on course data loaded from *data_path*."""
        df = load_courses(data_path)
        X, y = df, df["workload"]

        self._pipeline = self._build_pipeline()

        if cv:
            scores = cross_val_score(self._pipeline, X, y, cv=5, scoring="accuracy")
            print(
                f"Cross-validation accuracy: {scores.mean():.2f} "
                f"(± {scores.std():.2f})"
            )

        self._pipeline.fit(X, y)
        return self


    def predict(self, course_df: pd.DataFrame) -> list[str]:
        #Return predicted workload categories for each row in *course_df*.

        self._require_fitted()
        return self._pipeline.predict(course_df).tolist()  # type: ignore[union-attr]

    def predict_proba(self, course_df: pd.DataFrame) -> dict[str, float]:
        """Return class probabilities for the first row of *course_df*.

        Useful when we want a confidence breakdown rather than a single label.
        """
        self._require_fitted()
        proba = self._pipeline.predict_proba(course_df)[0]  # type: ignore[union-attr]
        classes: list[str] = self._pipeline.classes_.tolist()  # type: ignore[union-attr]
        return dict(zip(classes, proba))

    # Persistence

    def save(self, path: str | Path) -> None:
        """Serialise the fitted model to *path* using :mod:`joblib`.
        """
        self._require_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "WorkloadPredictor":
        """Deserialise a previously saved model.

        Parameters
        ----------
        path:
            Path to a ``.pkl`` / ``.joblib`` file created by :meth:`save`.

        Returns
        -------
        WorkloadPredictor
            A new instance with the restored pipeline.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        predictor = cls()
        predictor._pipeline = joblib.load(path)
        return predictor

    # Internal helpers

    def _build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("features", build_feature_pipeline()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        random_state=self.random_state,
                        class_weight="balanced",
                    )
                ),
            ]
        )

    def _require_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Model has not been fitted yet.  "
                "Call train() or load() first."
            )

    # Dunder helpers

    def __repr__(self) -> str:
        fitted = self._pipeline is not None
        return (
            f"WorkloadPredictor("
            f"n_estimators={self.n_estimators}, "
            f"random_state={self.random_state}, "
            f"fitted={fitted})"
        )
