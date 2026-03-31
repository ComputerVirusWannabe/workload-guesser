"""Data loading utilities for workload-guesser."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

#: Default path to the bundled sample dataset.
_SAMPLE_DATA_PATH = Path(__file__).parent.parent / "data" / "sample_courses.csv"

#: Required columns in any course DataFrame used for training.
REQUIRED_COLUMNS: list[str] = [
    "department",
    "level",
    "credits",
    "description",
    "workload",
]

#: Valid workload category labels.
WORKLOAD_LABELS: list[str] = ["low", "medium", "high"]


def load_courses(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load course data from *path*, defaulting to the bundled sample CSV.

    Parameters
    ----------
    path:
        Path to a CSV file.  If ``None`` the packaged sample dataset is used.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least the columns in :data:`REQUIRED_COLUMNS`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If any required column is missing or if the ``workload`` column contains
        values outside :data:`WORKLOAD_LABELS`.
    """
    csv_path = Path(path) if path is not None else _SAMPLE_DATA_PATH

    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if *df* is missing required columns or has bad labels."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid = set(df["workload"].dropna().unique()) - set(WORKLOAD_LABELS)
    if invalid:
        raise ValueError(
            f"Invalid workload label(s): {invalid}. "
            f"Allowed values: {WORKLOAD_LABELS}"
        )


def course_to_dataframe(
    department: str,
    level: int,
    credits: int,
    description: str,
    title: str = "",
    num_assignments: int = 0,
    num_exams: int = 0,
    num_projects: int = 0,
    gpa_avg: Optional[float] = None,
) -> pd.DataFrame:
    """Convert a single course's attributes into a one-row DataFrame.

    This is a convenience function for the CLI and API: it assembles the same
    column layout expected by the feature pipeline.
    """
    return pd.DataFrame(
        [
            {
                "department": department,
                "level": level,
                "credits": credits,
                "title": title,
                "description": description,
                "num_assignments": num_assignments,
                "num_exams": num_exams,
                "num_projects": num_projects,
                "gpa_avg": gpa_avg if gpa_avg is not None else 3.0,
            }
        ]
    )
