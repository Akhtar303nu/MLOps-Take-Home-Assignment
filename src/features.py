"""Feature engineering for the Telco churn model.

Single source of truth for both training and inference.
Using the same code path in both modes eliminates training-serving skew.

Design decision: a FittedFeatures dataclass carries the statistics computed
on training data (medians, scaler params). During inference, the caller must
pass in the FittedFeatures instance from the training run — never recompute
from live data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# The four features from the original notebook + the derived one.
# Explicit list here means the model always sees columns in the same order.
FEATURE_COLS = ["MonthlyCharges", "tenure", "TotalCharges", "HighValueFiber"]


@dataclass
class FittedFeatures:
    """Statistics fit on training data. Must be persisted alongside the model.

    Saved as an MLflow artifact so inference always uses training-time stats.
    """

    monthly_charges_median: float
    scaler_mean: list[float] = field(default_factory=list)
    scaler_scale: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "monthly_charges_median": self.monthly_charges_median,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FittedFeatures":
        return cls(
            monthly_charges_median=d["monthly_charges_median"],
            scaler_mean=d["scaler_mean"],
            scaler_scale=d["scaler_scale"],
        )


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the 11 rows in the original dataset where TotalCharges is whitespace."""
    df = df.copy()
    df = df[df["TotalCharges"].astype(str).str.strip() != ""]
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    return df


def _add_derived(df: pd.DataFrame, monthly_median: float) -> pd.DataFrame:
    """Add HighValueFiber flag.

    Args:
        df: Input dataframe.
        monthly_median: Must come from FittedFeatures in inference mode.
                        Prevents computing the median on live/partial data.
    """
    df = df.copy()
    df["HighValueFiber"] = (
        (df["InternetService"] == "Fiber optic") & (df["MonthlyCharges"] > monthly_median)
    ).astype(int)
    return df


def fit_and_transform(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, FittedFeatures]:
    """Fit feature pipeline on training data.

    Returns:
        X_scaled: Scaled feature matrix.
        y: Target array.
        fitted: Statistics to reuse at inference time.
    """
    df = _clean(df)

    # Fit median on training data only
    monthly_median = float(df["MonthlyCharges"].median())
    df = _add_derived(df, monthly_median)

    # Encode target
    y = df["Churn"].map({"Yes": 1, "No": 0}).values
    X = df[FEATURE_COLS].values

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    fitted = FittedFeatures(
        monthly_charges_median=monthly_median,
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
    )

    logger.info(
        "fit_and_transform: rows=%d churn_rate=%.2f%% median_monthly=%.2f",
        len(df),
        y.mean() * 100,
        monthly_median,
    )
    return X_scaled, y, fitted


def transform(df: pd.DataFrame, fitted: FittedFeatures) -> np.ndarray:
    """Apply pre-fitted pipeline to new data (inference mode).

    Args:
        df: Raw input dataframe.
        fitted: Statistics from the training run. Never recompute these.

    Returns:
        X_scaled: Scaled feature matrix ready for model.predict().
    """
    df = _clean(df)
    df = _add_derived(df, fitted.monthly_charges_median)
    X = df[FEATURE_COLS].values

    # Reconstruct scaler from saved params — no re-fitting
    scaler = StandardScaler()
    scaler.mean_ = np.array(fitted.scaler_mean)
    scaler.scale_ = np.array(fitted.scaler_scale)
    scaler.n_features_in_ = len(FEATURE_COLS)

    return scaler.transform(X)