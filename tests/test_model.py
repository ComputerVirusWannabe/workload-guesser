"""Tests for workload_guesser.model."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from workload_guesser.data import course_to_dataframe
from workload_guesser.model import WorkloadPredictor

# Fixtures

@pytest.fixture(scope="module")
def trained_predictor() -> WorkloadPredictor:
    """A predictor trained on the built-in sample data (shared across tests)."""
    predictor = WorkloadPredictor(n_estimators=50, random_state=0)
    predictor.train()
    return predictor

# Initialisation

class TestInit:
    def test_unfitted_repr(self) -> None:
        p = WorkloadPredictor()
        assert "fitted=False" in repr(p)

    def test_predict_before_train_raises(self) -> None:
        p = WorkloadPredictor()
        df = course_to_dataframe("CS", 3000, 3, "Some course.")
        with pytest.raises(RuntimeError, match="not been fitted"):
            p.predict(df)

    def test_predict_proba_before_train_raises(self) -> None:
        p = WorkloadPredictor()
        df = course_to_dataframe("CS", 3000, 3, "Some course.")
        with pytest.raises(RuntimeError, match="not been fitted"):
            p.predict_proba(df)


# Training
class TestTrain:
    def test_train_returns_self(self) -> None:
        p = WorkloadPredictor(n_estimators=10)
        result = p.train()
        assert result is p

    def test_fitted_repr_after_train(self) -> None:
        p = WorkloadPredictor(n_estimators=10)
        p.train()
        assert "fitted=True" in repr(p)

# Predict

class TestPredict:
    def test_returns_list(self, trained_predictor: WorkloadPredictor) -> None:
        df = course_to_dataframe("CS", 4000, 3, "Weekly exams and homework.")
        result = trained_predictor.predict(df)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_label_is_valid(self, trained_predictor: WorkloadPredictor) -> None:
        df = course_to_dataframe("HIST", 1000, 3, "Introductory survey. Three exams.")
        result = trained_predictor.predict(df)
        assert result[0] in {"low", "medium", "high"}

    def test_batch_prediction(self, trained_predictor: WorkloadPredictor, sample_df: pd.DataFrame) -> None:
        # Drop the target label column to simulate inference
        X = sample_df.drop(columns=["workload"])
        result = trained_predictor.predict(X)
        assert len(result) == len(sample_df)
        assert all(r in {"low", "medium", "high"} for r in result)

    def test_high_workload_course(self, trained_predictor: WorkloadPredictor) -> None:
        """A course with many exams/assignments should predict medium or high."""
        df = course_to_dataframe(
            department="CMSC",
            level=4000,
            credits=3,
            description=(
                "Rigorous course. Weekly homework assignments, "
                "weekly quizzes, two midterm exams, and a final exam. "
                "A demanding research project is required."
            ),
            gpa_avg=2.5,
        )
        result = trained_predictor.predict(df)
        assert result[0] in {"medium", "high"}

    def test_low_workload_course(self, trained_predictor: WorkloadPredictor) -> None:
        """A light activity/performance course should predict low or medium."""
        df = course_to_dataframe(
            department="KNES",
            level=1000,
            credits=1,
            description="Pass/fail activity course. Attendance only. No exams.",
            gpa_avg=4.0,
        )
        result = trained_predictor.predict(df)
        assert result[0] in {"low", "medium"}

# predict_proba


class TestPredictProba:
    def test_probabilities_sum_to_one(self, trained_predictor: WorkloadPredictor) -> None:
        df = course_to_dataframe("MATH", 3000, 4, "Weekly problem sets and two exams.")
        proba = trained_predictor.predict_proba(df)
        assert abs(sum(proba.values()) - 1.0) < 1e-6

    def test_keys_are_valid_labels(self, trained_predictor: WorkloadPredictor) -> None:
        df = course_to_dataframe("PSYC", 2000, 3, "Two exams and short essays.")
        proba = trained_predictor.predict_proba(df)
        assert set(proba.keys()) == {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# Persistence (save / load)
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load(self, trained_predictor: WorkloadPredictor, tmp_path: pathlib.Path) -> None:
        save_path = tmp_path / "model.pkl"
        trained_predictor.save(save_path)
        assert save_path.exists()

        loaded = WorkloadPredictor.load(save_path)
        df = course_to_dataframe("CS", 4000, 3, "Weekly exams and projects.")
        assert trained_predictor.predict(df) == loaded.predict(df)

    def test_load_nonexistent_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(FileNotFoundError):
            WorkloadPredictor.load(tmp_path / "no_such_file.pkl")

    def test_save_before_train_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(RuntimeError, match="not been fitted"):
            WorkloadPredictor().save(tmp_path / "model.pkl")
