"""UMD.io API integration for workload-guesser.

Provides helpers to fetch course data from the public UMD.io REST API
(https://api.umd.io/v1) and convert it into the DataFrame format expected
by :class:`~workload_guesser.model.WorkloadPredictor`.

Because UMD.io course records do not contain a ``workload`` label, the
output DataFrames produced here are suitable for **prediction only** — use
the bundled sample CSV (or your own labelled dataset) for training.

Typical usage::

    from workload_guesser.umd import fetch_course, umd_course_to_dataframe
    from workload_guesser.model import WorkloadPredictor

    predictor = WorkloadPredictor()
    predictor.train()

    course = fetch_course("CMSC131")
    df = umd_course_to_dataframe(course)
    print(predictor.predict(df))   # ['medium']
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd
import requests

#: Base URL for the UMD.io v1 REST API.
UMD_API_BASE = "https://api.umd.io/v1"

#: Default request timeout in seconds.
_TIMEOUT = 15


# ---------------------------------------------------------------------------
# API fetch helpers
# ---------------------------------------------------------------------------


def fetch_course(course_id: str, semester: Optional[str] = None) -> dict[str, Any]:
    """Fetch a single course by its *course_id* from the UMD API.

    Parameters
    ----------
    course_id:
        UMD course identifier, e.g. ``"CMSC131"`` or ``"MATH140"``.
    semester:
        Optional semester code in ``YYYYMM`` format, e.g. ``"202308"``
        for Fall 2023.  If omitted the API returns the most recent data.

    Returns
    -------
    dict
        Raw course object as returned by the API.

    Raises
    ------
    ValueError
        If the API returns an empty result for *course_id*.
    requests.HTTPError
        If the API returns a non-2xx status code.
    """
    params: dict[str, Any] = {}
    if semester:
        params["semester"] = semester

    url = f"{UMD_API_BASE}/courses/{course_id}"
    response = requests.get(url, params=params, timeout=_TIMEOUT)
    response.raise_for_status()

    data = response.json()
    if isinstance(data, list):
        if not data:
            raise ValueError(f"Course {course_id!r} not found in the UMD API.")
        return data[0]
    return data


def fetch_courses(
    dept_id: Optional[str] = None,
    semester: Optional[str] = None,
    per_page: int = 100,
    page: int = 1,
) -> list[dict[str, Any]]:
    """Fetch a page of courses from the UMD API.

    Parameters
    ----------
    dept_id:
        Optional four-letter department code to filter by, e.g. ``"CMSC"``.
    semester:
        Optional semester code (``YYYYMM``).
    per_page:
        Number of results per page (max 100).
    page:
        Page number for pagination (1-based).

    Returns
    -------
    list[dict]
        List of raw course objects.
    """
    params: dict[str, Any] = {"per_page": per_page, "page": page}
    if dept_id:
        params["dept_id"] = dept_id
    if semester:
        params["semester"] = semester

    url = f"{UMD_API_BASE}/courses"
    response = requests.get(url, params=params, timeout=_TIMEOUT)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _extract_level(course_id: str) -> int:
    """Return a normalised course level from a UMD *course_id*.

    UMD course numbers are three digits (e.g. ``131``, ``401``).  This
    function takes the leading digit and multiplies by 1 000 so that the
    result matches the 1000/2000/3000/4000 convention used in the training
    data (e.g. ``CMSC131`` → ``1000``, ``CMSC414`` → ``4000``).
    """
    match = re.search(r"(\d+)", course_id)
    if match:
        first_digit = int(match.group(1)[0])
        return first_digit * 1000
    return 1000  # safe fallback


def _parse_credits(credits_raw: Any) -> int:
    """Parse a UMD credits value into an integer.

    The API may return a string, an integer, or a range like ``"3-4"``.
    In range cases the lower bound is used.
    """
    try:
        return int(str(credits_raw).split("-")[0].strip())
    except (ValueError, TypeError):
        return 3  # neutral default


def umd_course_to_dataframe(course: dict[str, Any]) -> pd.DataFrame:
    """Convert a single UMD API course dict into a one-row prediction DataFrame.

    The returned DataFrame has the same column layout as
    :func:`~workload_guesser.data.course_to_dataframe` and is ready to be
    passed directly to :meth:`~workload_guesser.model.WorkloadPredictor.predict`.

    Note that the ``workload`` column is **not** present — this function is
    for inference, not training.

    Parameters
    ----------
    course:
        A raw course dict as returned by :func:`fetch_course` or
        :func:`fetch_courses`.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame suitable for workload prediction.
    """
    dept = course.get("dept_id") or course.get("department", "")
    course_id = course.get("course_id", "")
    level = _extract_level(course_id)
    credits_ = _parse_credits(course.get("credits", 3))
    description = course.get("description", "") or ""
    title = course.get("name", "") or ""

    return pd.DataFrame(
        [
            {
                "department": dept,
                "level": level,
                "credits": credits_,
                "title": title,
                "description": description,
                "num_assignments": 0,
                "num_exams": 0,
                "num_projects": 0,
                "gpa_avg": 3.0,
            }
        ]
    )


def umd_courses_to_dataframe(courses: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert a list of UMD API course dicts into a multi-row DataFrame.

    Useful for batch prediction over an entire department.

    Parameters
    ----------
    courses:
        List of raw course dicts (e.g. as returned by :func:`fetch_courses`).

    Returns
    -------
    pd.DataFrame
        Multi-row DataFrame ready for
        :meth:`~workload_guesser.model.WorkloadPredictor.predict`.
    """
    if not courses:
        return pd.DataFrame(
            columns=[
                "department",
                "level",
                "credits",
                "title",
                "description",
                "num_assignments",
                "num_exams",
                "num_projects",
                "gpa_avg",
            ]
        )
    return pd.concat(
        [umd_course_to_dataframe(c) for c in courses],
        ignore_index=True,
    )
