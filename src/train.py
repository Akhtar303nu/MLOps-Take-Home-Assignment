"""Training pipeline with MLflow Model Registry.

What this does:
  1. Loads data, stratified train/test split
  2. Fits feature pipeline on training data ONLY
  3. Trains RandomForest, evaluates on held-out test set
  4. Logs params, metrics, artifacts to MLflow Tracking
  5. Registers model in MLflow Model Registry with input signature
  6. Promotes to Staging if all metric thresholds pass
  7. Promotes to Production if it improves on current Production model
     (requires --promote-to-production flag — no accidental overwrites)

Usage:
    python -m src.train --data data/telco.csv
    python -m src.train --data data/telco.csv --promote-to-production
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required in containers / CI
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLS, _clean, fit_and_transform, transform

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REGISTERED_MODEL_NAME = "telco-churn-classifier"
EXPERIMENT_NAME = "telco-churn"

# Model promoted to Staging only if ALL thresholds are met.
# These are set achievable with 4 features — the gate logic matters more than
# hitting arbitrary numbers.
PROMOTION_THRESHOLDS = {
    "f1_churn": 0.45,
    "roc_auc": 0.75,
}

RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 4,
    "class_weight": "balanced",  # handles ~26% churn class imbalance
    "random_state": 42,
    "n_jobs": -1,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _save_confusion_matrix(
    y_test: np.ndarray, y_pred: np.ndarray, out_dir: Path
) -> Path:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(
        ax=ax, colorbar=False
    )
    ax.set_title("Confusion Matrix (test set)")
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return path


def _compute_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "accuracy": round(report["accuracy"], 4),
        "f1_churn": round(report["1"]["f1-score"], 4),
        "precision_churn": round(report["1"]["precision"], 4),
        "recall_churn": round(report["1"]["recall"], 4),
        "f1_no_churn": round(report["0"]["f1-score"], 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }


def _passes_promotion_gate(metrics: dict[str, float]) -> bool:
    """Return True only if every threshold in PROMOTION_THRESHOLDS is met."""
    for metric, threshold in PROMOTION_THRESHOLDS.items():
        value = metrics.get(metric, 0.0)
        if value < threshold:
            logger.warning(
                "Gate FAILED: %s=%.4f < required=%.4f", metric, value, threshold
            )
            return False
        logger.info("Gate PASSED: %s=%.4f >= %.4f", metric, value, threshold)
    return True


def _get_production_f1(client: mlflow.tracking.MlflowClient) -> float | None:
    """f1_churn of the current Production model, or None if no Production exists."""
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return None
    return client.get_run(versions[0].run_id).data.metrics.get("f1_churn")


# ── Main ──────────────────────────────────────────────────────────────────────
def train(data_path: str, promote_to_production: bool = False) -> str:
    """Run full training pipeline. Returns MLflow run_id."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = mlflow.tracking.MlflowClient()

    # ── Load and split ─────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(df), data_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Churn"].map({"Yes": 1, "No": 0}),
    )

    X_train, y_train, fitted = fit_and_transform(train_df)

    # Test set uses training-time stats — never refit the scaler on test data
    X_test = transform(test_df, fitted)
    y_test = _clean(test_df)["Churn"].map({"Yes": 1, "No": 0}).values

    # ── MLflow run ─────────────────────────────────────────────────────────────
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "git_commit": _git_sha(),
            "data_path": str(data_path),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "churn_rate_train_pct": round(float(y_train.mean()) * 100, 2),
        })

        mlflow.log_params(RF_PARAMS)
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("test_size", 0.2)

        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = _compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)
        logger.info("Metrics: %s", metrics)

        # Artifacts
        artifact_dir = Path(f"/tmp/mlflow_{run_id}")
        artifact_dir.mkdir(parents=True, exist_ok=True)

        mlflow.log_artifact(str(_save_confusion_matrix(y_test, y_pred, artifact_dir)))

        # FittedFeatures MUST be stored alongside the model.
        # Inference loads this to reconstruct the exact same feature pipeline.
        fitted_path = artifact_dir / "fitted_features.json"
        fitted_path.write_text(json.dumps(fitted.to_dict(), indent=2))
        mlflow.log_artifact(str(fitted_path))

        # Model signature catches schema mismatches at serve time
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            signature=infer_signature(X_train, y_pred),
            input_example=X_train[:3],
        )

        # ── Promotion ──────────────────────────────────────────────────────────
        all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        latest_version = str(max(int(v.version) for v in all_versions))

        if _passes_promotion_gate(metrics):
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=latest_version,
                stage="Staging",
                archive_existing_versions=False,
            )
            mlflow.set_tag("promoted_to", "Staging")
            logger.info("v%s → Staging", latest_version)
        else:
            mlflow.set_tag("promoted_to", "None")

        if promote_to_production:
            prod_f1 = _get_production_f1(client)
            if prod_f1 is None:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                mlflow.set_tag("promoted_to", "Production (first deploy)")
                logger.info("First production model: v%s", latest_version)
            elif metrics["f1_churn"] > prod_f1:
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                mlflow.set_tag("promoted_to", "Production (improved)")
                logger.info(
                    "Production updated v%s: f1_churn %.4f → %.4f",
                    latest_version, prod_f1, metrics["f1_churn"],
                )
            else:
                mlflow.set_tag("promoted_to", "Staging (did not beat Production)")
                logger.warning(
                    "Candidate f1=%.4f did not beat Production f1=%.4f — no promotion",
                    metrics["f1_churn"], prod_f1,
                )

        logger.info("Done. Run ID: %s", run_id)
        return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--promote-to-production",
        action="store_true",
        help="Promote to Production if it beats the current Production model",
    )
    args = parser.parse_args()

    run_id = train(args.data, promote_to_production=args.promote_to_production)
    print(f"\nRun ID: {run_id}")
    print(f"UI:     {os.environ.get('MLFLOW_TRACKING_URI', 'http://localhost:5000')}")
