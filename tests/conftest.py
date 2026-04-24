"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from workload_guesser.data import course_to_dataframe


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A tiny DataFrame representative of UMD courses at each workload level."""
    return pd.DataFrame(
        [
            {
                "department": "KNES",
                "level": 1000,
                "credits": 1,
                "description": "Pass/fail attendance-based activity course. No exams or assignments.",
                "workload": "low",
                "gpa_avg": 4.0,
            },
            {
                "department": "ECON",
                "level": 2000,
                "credits": 3,
                "description": "Three exams and weekly readings. Minimal written homework.",
                "workload": "low",
                "gpa_avg": 3.4,
            },
            {
                "department": "MATH",
                "level": 2000,
                "credits": 4,
                "description": "Weekly homework, two midterm exams and a final exam.",
                "workload": "medium",
                "gpa_avg": 3.0,
            },
            {
                "department": "CMSC",
                "level": 4000,
                "credits": 3,
                "description": (
                    "Rigorous weekly problem sets, two midterms, final exam, "
                    "and a semester-long project."
                ),
                "workload": "high",
                "gpa_avg": 2.7,
            },
        ]
    )


@pytest.fixture()
def single_course_df() -> pd.DataFrame:
    """A one-row DataFrame for a single high-workload UMD course."""
    return course_to_dataframe(
        department="CMSC",
        level=4000,
        credits=3,
        description=(
            "Challenging algorithms course with weekly problem sets, "
            "two midterm exams, and a final exam."
        ),
        gpa_avg=2.6,
    )
