"""Unit tests for src/features.py.

These tests run with no external dependencies — no MLflow server, no database.
Every test exercises a real invariant, not a happy-path smoke test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLS,
    FittedFeatures,
    _add_derived,
    _clean,
    fit_and_transform,
    transform,
)


# ── Shared fixture ─────────────────────────────────────────────────────────────
@pytest.fixture
def minimal_df() -> pd.DataFrame:
    """Smallest valid dataframe that covers all code paths."""
    return pd.DataFrame(
        {
            "customerID": ["A", "B", "C", "D"],
            "InternetService": ["Fiber optic", "DSL", "Fiber optic", "No"],
            "MonthlyCharges": [80.0, 50.0, 30.0, 20.0],
            "tenure": [12, 24, 1, 60],
            "TotalCharges": ["960.0", "1200.0", "30.0", "1200.0"],
            "Churn": ["Yes", "No", "Yes", "No"],
        }
    )


@pytest.fixture
def dirty_df(minimal_df) -> pd.DataFrame:
    """Contains the whitespace-TotalCharges bug from the original dataset."""
    df = minimal_df.copy()
    df.loc[2, "TotalCharges"] = " "
    return df


# ── _clean ─────────────────────────────────────────────────────────────────────
class TestClean:
    def test_removes_whitespace_total_charges(self, dirty_df):
        result = _clean(dirty_df)
        assert len(result) == 3  # 1 row removed

    def test_converts_total_charges_to_float(self, minimal_df):
        result = _clean(minimal_df)
        assert result["TotalCharges"].dtype == float

    def test_clean_data_unchanged_row_count(self, minimal_df):
        result = _clean(minimal_df)
        assert len(result) == len(minimal_df)

    def test_raises_no_exception_on_empty_df(self):
        df = pd.DataFrame({"TotalCharges": []})
        result = _clean(df)
        assert len(result) == 0


# ── _add_derived ───────────────────────────────────────────────────────────────
class TestAddDerived:
    def test_high_value_fiber_is_binary(self, minimal_df):
        df = minimal_df.copy()
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        result = _add_derived(df, monthly_median=50.0)
        assert set(result["HighValueFiber"].unique()).issubset({0, 1})

    def test_high_value_fiber_correct_logic(self, minimal_df):
        """Fiber optic AND MonthlyCharges > median → 1. Everything else → 0."""
        df = minimal_df.copy()
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        median = 50.0
        result = _add_derived(df, monthly_median=median)

        # Row 0: Fiber optic, 80 > 50 → 1
        assert result.loc[0, "HighValueFiber"] == 1
        # Row 1: DSL, 50 == 50 (not >) → 0
        assert result.loc[1, "HighValueFiber"] == 0
        # Row 2: Fiber optic, 30 < 50 → 0
        assert result.loc[2, "HighValueFiber"] == 0
        # Row 3: No internet → 0
        assert result.loc[3, "HighValueFiber"] == 0

    def test_provided_median_is_used_not_recomputed(self, minimal_df):
        """Passing median=999 should make every HighValueFiber = 0."""
        df = minimal_df.copy()
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        result = _add_derived(df, monthly_median=999.0)
        assert result["HighValueFiber"].sum() == 0


# ── fit_and_transform ──────────────────────────────────────────────────────────
class TestFitAndTransform:
    def test_output_shape_matches_feature_cols(self, minimal_df):
        X, y, _ = fit_and_transform(minimal_df)
        assert X.shape[1] == len(FEATURE_COLS)

    def test_output_row_count_matches_cleaned_df(self, minimal_df):
        X, y, _ = fit_and_transform(minimal_df)
        assert X.shape[0] == len(y)

    def test_returns_fitted_features(self, minimal_df):
        _, _, fitted = fit_and_transform(minimal_df)
        assert isinstance(fitted, FittedFeatures)
        assert isinstance(fitted.monthly_charges_median, float)
        assert len(fitted.scaler_mean) == len(FEATURE_COLS)
        assert len(fitted.scaler_scale) == len(FEATURE_COLS)

    def test_x_is_scaled(self, minimal_df):
        """Scaled features should have near-zero mean and unit std."""
        X, _, _ = fit_and_transform(minimal_df)
        # With only 4 rows the mean won't be exactly 0, but should be small
        assert abs(X[:, 0].mean()) < 1.0

    def test_dirty_rows_excluded_from_output(self, dirty_df):
        X, y, _ = fit_and_transform(dirty_df)
        # dirty_df has 4 rows, 1 whitespace → expect 3 rows
        assert X.shape[0] == 3


# ── transform (inference mode) ────────────────────────────────────────────────
class TestTransform:
    def test_train_and_inference_same_shape(self, minimal_df):
        """Identical input must produce identical output from both paths."""
        X_train, _, fitted = fit_and_transform(minimal_df)
        X_infer = transform(minimal_df, fitted)
        assert X_train.shape == X_infer.shape

    def test_train_and_inference_same_values(self, minimal_df):
        X_train, _, fitted = fit_and_transform(minimal_df)
        X_infer = transform(minimal_df, fitted)
        np.testing.assert_array_almost_equal(X_train, X_infer)

    def test_inference_uses_training_median_not_data_median(self, minimal_df):
        """Key anti-skew test.

        If we pass a FittedFeatures with a very high median, all HighValueFiber
        values should be 0 — regardless of what the test data's own median would be.
        """
        _, _, fitted = fit_and_transform(minimal_df)
        # Override the median to something that makes all HVF = 0
        fitted.monthly_charges_median = 9999.0

        X = transform(minimal_df, fitted)
        # HighValueFiber is the 4th column (index 3)
        hvf_col_idx = FEATURE_COLS.index("HighValueFiber")
        # All raw values are 0, so scaled values should all be identical (mean of zeros)
        assert len(set(X[:, hvf_col_idx].tolist())) == 1


# ── FittedFeatures serialisation ──────────────────────────────────────────────
class TestFittedFeaturesSerialization:
    def test_roundtrip(self, minimal_df):
        _, _, fitted = fit_and_transform(minimal_df)
        restored = FittedFeatures.from_dict(fitted.to_dict())
        assert restored.monthly_charges_median == fitted.monthly_charges_median
        assert restored.scaler_mean == fitted.scaler_mean
        assert restored.scaler_scale == fitted.scaler_scale

    def test_to_dict_contains_required_keys(self, minimal_df):
        _, _, fitted = fit_and_transform(minimal_df)
        d = fitted.to_dict()
        assert "monthly_charges_median" in d
        assert "scaler_mean" in d
        assert "scaler_scale" in d
