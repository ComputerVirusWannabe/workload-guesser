"""Tests for workload_guesser.data."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from workload_guesser.data import (
    WORKLOAD_LABELS,
    course_to_dataframe,
    load_courses,
)


class TestLoadCourses:
    def test_loads_sample_data_by_default(self) -> None:
        df = load_courses()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self) -> None:
        df = load_courses()
        for col in ("department", "level", "credits", "description", "workload"):
            assert col in df.columns, f"Missing column: {col}"

    def test_workload_labels_are_valid(self) -> None:
        df = load_courses()
        invalid = set(df["workload"].unique()) - set(WORKLOAD_LABELS)
        assert not invalid, f"Invalid workload labels: {invalid}"

    def test_all_three_labels_present(self) -> None:
        df = load_courses()
        labels = set(df["workload"].unique())
        assert labels == {"low", "medium", "high"}

    def test_file_not_found_raises(self, tmp_path: "pathlib.Path") -> None:
        with pytest.raises(FileNotFoundError):
            load_courses(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path: "pathlib.Path") -> None:
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("department,level\nCS,1000\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_courses(bad_csv)

    def test_invalid_workload_label_raises(self, tmp_path: "pathlib.Path") -> None:
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text(
            "department,level,credits,description,workload\n"
            "CS,1000,3,Some description,extreme\n"
        )
        with pytest.raises(ValueError, match="Invalid workload label"):
            load_courses(bad_csv)


class TestCourseToDataframe:
    def test_returns_single_row(self) -> None:
        df = course_to_dataframe("CS", 4000, 3, "Weekly homework and exams.")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_columns_present(self) -> None:
        df = course_to_dataframe("MATH", 3000, 4, "Proof-based weekly homework.")
        for col in ("department", "level", "credits", "description"):
            assert col in df.columns

    def test_gpa_default_is_3(self) -> None:
        df = course_to_dataframe("HIST", 2000, 3, "Lectures and readings.")
        assert df["gpa_avg"].iloc[0] == 3.0

    def test_gpa_explicit(self) -> None:
        df = course_to_dataframe("CS", 4000, 3, "Hard course.", gpa_avg=2.5)
        assert df["gpa_avg"].iloc[0] == 2.5
