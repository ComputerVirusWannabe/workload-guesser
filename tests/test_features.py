"""Tests for workload_guesser.features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from workload_guesser.features import (
    KeywordCountTransformer,
    MetadataTransformer,
    TextSelector,
    build_feature_pipeline,
    build_text_pipeline,
)


# ---------------------------------------------------------------------------
# TextSelector
# ---------------------------------------------------------------------------


class TestTextSelector:
    def test_returns_list_of_strings(self, sample_df: pd.DataFrame) -> None:
        selector = TextSelector(column="description")
        result = selector.fit_transform(sample_df)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    def test_length_matches_input_rows(self, sample_df: pd.DataFrame) -> None:
        selector = TextSelector(column="description")
        result = selector.fit_transform(sample_df)
        assert len(result) == len(sample_df)

    def test_missing_values_become_empty_string(self) -> None:
        df = pd.DataFrame({"description": [None, "hello", np.nan]})
        selector = TextSelector(column="description")
        result = selector.fit_transform(df)
        assert result[0] == ""
        assert result[2] == ""


# ---------------------------------------------------------------------------
# KeywordCountTransformer
# ---------------------------------------------------------------------------


class TestKeywordCountTransformer:
    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        transformer = KeywordCountTransformer()
        result = transformer.fit_transform(sample_df)
        assert result.shape[0] == len(sample_df)
        assert result.shape[1] == len(transformer.keywords)

    def test_counts_keyword_correctly(self) -> None:
        df = pd.DataFrame({"description": ["exam exam midterm quiz assignment"]})
        transformer = KeywordCountTransformer(keywords=["exam", "midterm", "assignment"])
        result = transformer.fit_transform(df)
        assert result[0, 0] == 2  # 'exam' appears twice
        assert result[0, 1] == 1  # 'midterm' once
        assert result[0, 2] == 1  # 'assignment' once

    def test_case_insensitive(self) -> None:
        df = pd.DataFrame({"description": ["EXAM Exam exam"]})
        transformer = KeywordCountTransformer(keywords=["exam"])
        result = transformer.fit_transform(df)
        assert result[0, 0] == 3

    def test_no_false_matches_on_substrings(self) -> None:
        """'examinable' and 'examination' should NOT count as 'exam'."""
        df = pd.DataFrame({"description": ["examinable examination"]})
        transformer = KeywordCountTransformer(keywords=["exam"])
        result = transformer.fit_transform(df)
        assert result[0, 0] == 0


# ---------------------------------------------------------------------------
# MetadataTransformer
# ---------------------------------------------------------------------------


class TestMetadataTransformer:
    def test_output_shape(self, sample_df: pd.DataFrame) -> None:
        meta = MetadataTransformer()
        result = meta.fit_transform(sample_df)
        assert result.shape == (len(sample_df), 3)

    def test_level_normalised(self) -> None:
        df = pd.DataFrame({"level": [2000], "credits": [3], "gpa_avg": [3.0]})
        meta = MetadataTransformer()
        result = meta.fit_transform(df)
        assert result[0, 0] == pytest.approx(2.0)

    def test_missing_gpa_defaults_to_3(self) -> None:
        df = pd.DataFrame({"level": [1000], "credits": [3], "gpa_avg": [None]})
        meta = MetadataTransformer()
        result = meta.fit_transform(df)
        assert result[0, 2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# build_text_pipeline / build_feature_pipeline
# ---------------------------------------------------------------------------


class TestBuildTextPipeline:
    def test_produces_sparse_matrix(self, sample_df: pd.DataFrame) -> None:
        from scipy.sparse import issparse

        pipe = build_text_pipeline()
        out = pipe.fit_transform(sample_df)
        assert issparse(out)

    def test_n_rows_matches_input(self, sample_df: pd.DataFrame) -> None:
        pipe = build_text_pipeline()
        out = pipe.fit_transform(sample_df)
        assert out.shape[0] == len(sample_df)


class TestBuildFeaturePipeline:
    def test_output_is_array_like(self, sample_df: pd.DataFrame) -> None:
        from scipy.sparse import issparse

        pipe = build_feature_pipeline()
        out = pipe.fit_transform(sample_df)
        # FeatureUnion may return a dense ndarray or sparse CSR depending on
        # whether the TF-IDF transformer dominates; both are acceptable inputs
        # for sklearn estimators.
        assert isinstance(out, np.ndarray) or issparse(out)

    def test_n_rows_matches_input(self, sample_df: pd.DataFrame) -> None:
        pipe = build_feature_pipeline()
        out = pipe.fit_transform(sample_df)
        assert out.shape[0] == len(sample_df)

    def test_no_nans_in_output(self, sample_df: pd.DataFrame) -> None:
        from scipy.sparse import issparse

        pipe = build_feature_pipeline()
        out = pipe.fit_transform(sample_df)
        arr = out.toarray() if issparse(out) else out
        assert not np.isnan(arr).any()
