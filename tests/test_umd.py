"""Tests for workload_guesser.umd."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from workload_guesser.umd import (
    _extract_level,
    _parse_credits,
    umd_course_to_dataframe,
    umd_courses_to_dataframe,
)


# ---------------------------------------------------------------------------
# _extract_level
# ---------------------------------------------------------------------------


class TestExtractLevel:
    def test_100_level(self) -> None:
        assert _extract_level("CMSC131") == 1000

    def test_200_level(self) -> None:
        assert _extract_level("MATH240") == 2000

    def test_300_level(self) -> None:
        assert _extract_level("ENGL301") == 3000

    def test_400_level(self) -> None:
        assert _extract_level("CMSC414") == 4000

    def test_600_level(self) -> None:
        assert _extract_level("CMSC660") == 6000

    def test_no_digits_returns_1000(self) -> None:
        assert _extract_level("NODIGS") == 1000

    def test_course_id_with_letter_suffix(self) -> None:
        # e.g. MATH141H
        assert _extract_level("MATH141H") == 1000


# ---------------------------------------------------------------------------
# _parse_credits
# ---------------------------------------------------------------------------


class TestParseCredits:
    def test_integer_string(self) -> None:
        assert _parse_credits("3") == 3

    def test_plain_integer(self) -> None:
        assert _parse_credits(4) == 4

    def test_range_string_lower_bound(self) -> None:
        assert _parse_credits("3-4") == 3

    def test_invalid_falls_back_to_3(self) -> None:
        assert _parse_credits("TBD") == 3

    def test_none_falls_back_to_3(self) -> None:
        assert _parse_credits(None) == 3


# ---------------------------------------------------------------------------
# umd_course_to_dataframe
# ---------------------------------------------------------------------------


class TestUmdCourseToDataframe:
    _SAMPLE_COURSE: dict = {
        "course_id": "CMSC351",
        "name": "Algorithms",
        "dept_id": "CMSC",
        "department": "Computer Science",
        "credits": "3",
        "description": "Rigorous study of algorithms and complexity.",
        "grading_method": ["Regular"],
        "gen_ed": [],
        "sections": [],
    }

    def test_returns_single_row_dataframe(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_department_from_dept_id(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["department"].iloc[0] == "CMSC"

    def test_level_extracted_correctly(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["level"].iloc[0] == 3000

    def test_credits_parsed(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["credits"].iloc[0] == 3

    def test_description_preserved(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["description"].iloc[0] == self._SAMPLE_COURSE["description"]

    def test_title_preserved(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["title"].iloc[0] == "Algorithms"

    def test_no_workload_column(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert "workload" not in df.columns

    def test_gpa_avg_defaults_to_3(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["gpa_avg"].iloc[0] == 3.0

    def test_assignment_counts_default_to_zero(self) -> None:
        df = umd_course_to_dataframe(self._SAMPLE_COURSE)
        assert df["num_assignments"].iloc[0] == 0
        assert df["num_exams"].iloc[0] == 0
        assert df["num_projects"].iloc[0] == 0

    def test_missing_description_becomes_empty_string(self) -> None:
        course = {**self._SAMPLE_COURSE, "description": None}
        df = umd_course_to_dataframe(course)
        assert df["description"].iloc[0] == ""

    def test_falls_back_to_department_field_when_dept_id_absent(self) -> None:
        course = {k: v for k, v in self._SAMPLE_COURSE.items() if k != "dept_id"}
        df = umd_course_to_dataframe(course)
        assert df["department"].iloc[0] == "Computer Science"

    def test_credit_range_uses_lower_bound(self) -> None:
        course = {**self._SAMPLE_COURSE, "credits": "3-4"}
        df = umd_course_to_dataframe(course)
        assert df["credits"].iloc[0] == 3


# ---------------------------------------------------------------------------
# umd_courses_to_dataframe
# ---------------------------------------------------------------------------


class TestUmdCoursesToDataframe:
    _COURSES = [
        {
            "course_id": "CMSC131",
            "name": "OOP I",
            "dept_id": "CMSC",
            "credits": "4",
            "description": "Intro to Java.",
        },
        {
            "course_id": "MATH140",
            "name": "Calculus I",
            "dept_id": "MATH",
            "credits": "4",
            "description": "Limits and derivatives.",
        },
    ]

    def test_returns_dataframe(self) -> None:
        df = umd_courses_to_dataframe(self._COURSES)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_input(self) -> None:
        df = umd_courses_to_dataframe(self._COURSES)
        assert len(df) == 2

    def test_empty_list_returns_empty_dataframe(self) -> None:
        df = umd_courses_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_departments_correct(self) -> None:
        df = umd_courses_to_dataframe(self._COURSES)
        assert list(df["department"]) == ["CMSC", "MATH"]

    def test_levels_extracted(self) -> None:
        df = umd_courses_to_dataframe(self._COURSES)
        assert list(df["level"]) == [1000, 1000]

    def test_no_workload_column(self) -> None:
        df = umd_courses_to_dataframe(self._COURSES)
        assert "workload" not in df.columns


# ---------------------------------------------------------------------------
# fetch_course (mocked)
# ---------------------------------------------------------------------------


class TestFetchCourse:
    def test_returns_dict_from_list_response(self) -> None:
        from workload_guesser.umd import fetch_course

        mock_response = MagicMock()
        mock_response.json.return_value = [{"course_id": "CMSC351", "name": "Algorithms"}]
        mock_response.raise_for_status.return_value = None

        with patch("workload_guesser.umd.requests.get", return_value=mock_response):
            result = fetch_course("CMSC351")

        assert result["course_id"] == "CMSC351"

    def test_returns_dict_directly_when_not_list(self) -> None:
        from workload_guesser.umd import fetch_course

        mock_response = MagicMock()
        mock_response.json.return_value = {"course_id": "CMSC351", "name": "Algorithms"}
        mock_response.raise_for_status.return_value = None

        with patch("workload_guesser.umd.requests.get", return_value=mock_response):
            result = fetch_course("CMSC351")

        assert result["course_id"] == "CMSC351"

    def test_raises_value_error_on_empty_list(self) -> None:
        from workload_guesser.umd import fetch_course

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        with patch("workload_guesser.umd.requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="not found"):
                fetch_course("FAKE999")

    def test_semester_param_passed_to_request(self) -> None:
        from workload_guesser.umd import fetch_course

        mock_response = MagicMock()
        mock_response.json.return_value = [{"course_id": "CMSC131"}]
        mock_response.raise_for_status.return_value = None

        with patch("workload_guesser.umd.requests.get", return_value=mock_response) as mock_get:
            fetch_course("CMSC131", semester="202308")

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["semester"] == "202308"


# ---------------------------------------------------------------------------
# fetch_courses (mocked)
# ---------------------------------------------------------------------------


class TestFetchCourses:
    def test_returns_list(self) -> None:
        from workload_guesser.umd import fetch_courses

        mock_response = MagicMock()
        mock_response.json.return_value = [{"course_id": "CMSC131"}, {"course_id": "CMSC132"}]
        mock_response.raise_for_status.return_value = None

        with patch("workload_guesser.umd.requests.get", return_value=mock_response):
            result = fetch_courses(dept_id="CMSC")

        assert isinstance(result, list)
        assert len(result) == 2

    def test_dept_id_passed_to_request(self) -> None:
        from workload_guesser.umd import fetch_courses

        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None

        with patch("workload_guesser.umd.requests.get", return_value=mock_response) as mock_get:
            fetch_courses(dept_id="MATH", per_page=50)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["dept_id"] == "MATH"
        assert kwargs["params"]["per_page"] == 50


# ---------------------------------------------------------------------------
# Integration: umd_course_to_dataframe feeds into WorkloadPredictor
# ---------------------------------------------------------------------------


class TestUmdIntegrationWithPredictor:
    def test_prediction_runs_on_umd_dataframe(self) -> None:
        from workload_guesser.model import WorkloadPredictor

        predictor = WorkloadPredictor(n_estimators=10, random_state=0)
        predictor.train()

        course = {
            "course_id": "CMSC351",
            "name": "Algorithms",
            "dept_id": "CMSC",
            "credits": "3",
            "description": (
                "Rigorous study of algorithm design and analysis. "
                "Weekly problem sets, a midterm exam, and a final exam."
            ),
        }
        df = umd_course_to_dataframe(course)
        result = predictor.predict(df)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] in {"low", "medium", "high"}
