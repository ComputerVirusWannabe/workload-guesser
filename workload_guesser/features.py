"""Feature extraction for course workload prediction.

Two groups of features are produced:

1. **Text features** – extracted from the free-text course *description* using
   TF-IDF weighted uni- and bi-grams, augmented with hand-crafted keyword
   counts that directly signal workload intensity (exams, assignments, etc.).

2. **Metadata features** – numerical signals derived from structured course
   attributes: credit hours, course level (normalised to the 1-8 range), and
   average historical GPA (a proxy for grading difficulty).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

#: Words / phrases whose presence in a description suggests heavier workload.
WORKLOAD_KEYWORDS: list[str] = [
    "assignment",
    "homework",
    "problem set",
    "weekly",
    "project",
    "paper",
    "essay",
    "report",
    "lab",
    "exam",
    "quiz",
    "test",
    "midterm",
    "final",
    "presentation",
    "reading",
    "rigorous",
    "intensive",
    "demanding",
    "challenging",
    "research",
]


# ---------------------------------------------------------------------------
# Custom sklearn transformers
# ---------------------------------------------------------------------------


class TextSelector(BaseEstimator, TransformerMixin):
    """Select a single text column from a DataFrame and return it as a Series."""

    def __init__(self, column: str = "description") -> None:
        self.column = column

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TextSelector":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> list[str]:  # noqa: N803
        return X[self.column].fillna("").tolist()


class KeywordCountTransformer(BaseEstimator, TransformerMixin):
    """Count occurrences of workload-related keywords in a text column.

    Returns a dense NumPy array of shape ``(n_samples, n_keywords)``.
    """

    def __init__(
        self,
        column: str = "description",
        keywords: list[str] | None = None,
    ) -> None:
        self.column = column
        self.keywords = keywords if keywords is not None else WORKLOAD_KEYWORDS

    def fit(self, X: pd.DataFrame, y: Any = None) -> "KeywordCountTransformer":  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        texts = X[self.column].fillna("").str.lower().tolist()
        result = np.zeros((len(texts), len(self.keywords)), dtype=float)
        for i, text in enumerate(texts):
            for j, kw in enumerate(self.keywords):
                pattern = r"\b" + re.escape(kw) + r"\b"
                result[i, j] = len(re.findall(pattern, text))
        return result


class MetadataTransformer(BaseEstimator, TransformerMixin):
    """Extract and scale numerical course metadata features.

    Features produced:
    * ``course_level_norm`` – course level divided by 1000 (range 1–8).
    * ``credits`` – credit hours as-is.
    * ``gpa_avg`` – average historical GPA (0–4 scale); missing values are
      imputed with 3.0 (a neutral, B-average assumption).
    * ``num_assignments`` – number of assignments
    * ``num_exams`` – number of exams
    * ``num_projects`` – number of projects
    """

    #: Columns expected (and their fallback default values).
    _COLUMNS: dict[str, float] = {
        "level": 1000.0,
        "credits": 3.0,
        "gpa_avg": 3.0,
        "num_assignments": 0,
        "num_exams": 0,
        "num_projects": 0,
    }

    def fit(self, X: pd.DataFrame, y: Any = None) -> "MetadataTransformer":  
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        rows: list[list[float]] = []
        for _, row in X.iterrows():
            level = float(row.get("level", self._COLUMNS["level"]) or self._COLUMNS["level"])
            credits_ = float(row.get("credits", self._COLUMNS["credits"]) or self._COLUMNS["credits"])
            gpa = float(row.get("gpa_avg", self._COLUMNS["gpa_avg"]) or self._COLUMNS["gpa_avg"])
            num_assignments = float(row.get("num_assignments", 0))
            num_exams = float(row.get("num_exams", 0))
            num_projects = float(row.get("num_projects", 0))

            rows.append([
                level / 1000.0,
                credits_,
                gpa,
                num_assignments,
                num_exams,
                num_projects,
            ])
        return np.array(rows, dtype=float)


# Pipeline builder


def build_text_pipeline() -> Pipeline:
    """Return a sklearn Pipeline that converts a DataFrame into TF-IDF features."""
    return Pipeline(
        [
            ("selector", TextSelector(column="description")),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=500,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
        ]
    )


def build_feature_pipeline() -> FeatureUnion:
    """Return a ``FeatureUnion`` combining TF-IDF, keyword counts, and metadata."""
    return FeatureUnion(
        [
            ("tfidf", build_text_pipeline()),
            (
                "keywords",
                Pipeline(
                    [
                        ("kw_counts", KeywordCountTransformer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
            (
                "metadata",
                Pipeline(
                    [
                        ("meta", MetadataTransformer()),
                        ("scaler", StandardScaler()),
                    ]
                ),
            ),
        ]
    )
