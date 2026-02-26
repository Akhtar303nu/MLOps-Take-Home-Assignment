# Telco Churn — MLOps Take-Home Assessment

## Component implemented: Model Registry

**Why Model Registry?**

The registry is the single source of truth that every other MLOps component
depends on. Monitoring needs to know which model version is in production.
Retraining needs somewhere to register the new version and compare it against
the current one. Deployment needs to pull a specific, versioned artifact. None
of the other components work correctly without a registry — so this is the right
starting point.

---

## What was built

```
src/features.py    — Feature engineering. Same function used for training and inference.
src/train.py       — Trains a GradientBoostingClassifier, logs everything to MLflow,
                     registers the model, promotes to Staging/Production based on metric gates.
src/serve.py       — FastAPI inference endpoint. Loads Production model + FittedFeatures
                     from MLflow at startup. Exposes /predict and /health.
tests/             — 17 unit tests that verify the feature pipeline correctness,
                     including the training-serving skew test.
docker-compose.yaml — MLflow server + PostgreSQL (metadata) + MinIO/S3 (artifacts).
Dockerfile         — Training image (python:3.11-slim).
Dockerfile.serve   — Inference image. Separate from training — no training dependencies.
.github/workflows/ — CI: lint (flake8 + black) + test + docker build on every PR.
```

## The one design decision worth explaining

The `FittedFeatures` dataclass in `features.py` carries the statistics computed
on training data (the monthly charges median, the scaler mean and scale). These
are logged as a JSON artifact alongside the model in MLflow.

When serving, you load `FittedFeatures` from MLflow — you never recompute the
median or refit the scaler from live data. This prevents training-serving skew.

The test `test_inference_uses_training_medians` verifies this explicitly: it
fits on a training split, then runs inference on a held-out split using the
frozen statistics and asserts there are no nulls and column order is preserved,
regardless of what the live data's actual distribution would produce.

---

## Prerequisites

- **Python 3.11** — install via `brew install python@3.11` (locally tested on 3.11.14)
- **Docker & Docker Compose**
- **Make**

---

## Local Setup

```bash
git clone <repo>
cd telco-churn-mlops
cp .env.example .env

# Create virtual environment with Python 3.11 (tested on 3.11.14)
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

# Verify Python version
python --version  # must show Python 3.11.x

# Install dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt

# Verify all tests pass
pytest tests/ -v  # 17/17 expected
```

## Environment

| Variable | Value |
|----------|-------|
| Python | 3.11.14 |
| Platform | ARM Mac (Apple Silicon) / Linux amd64 |
| Virtual environment | `.venv` (python3.11 -m venv) |
| Package manager | pip |

---

## Run

### Option A — Full stack (Docker)
Postgres metadata + MinIO artifact store (production-grade)

```bash
make stack-up
make train
```

### Option B — Local dev (no Docker)
SQLite metadata + local filesystem artifacts

```bash
make dev
make train
```

### Other targets

```bash
make test          # run tests with coverage
make lint          # flake8 + black check
make train-promote # promote to Production if model beats current prod
make stack-down    # tear down stack and volumes
```

## MLflow UI

Open http://localhost:5000 after `make stack-up` or `make dev`.

You will see:

- **Experiments** → `telco-churn-training` → each run with params, metrics, and `fitted_features.json` artifact
- **Models** → `telco-churn` → version history with Staging/Production/Archived stages

---

## Promotion logic

A model is only promoted to **Staging** if it passes all thresholds:
`f1_churn >= 0.45` and `roc_auc >= 0.75`.

A model is only promoted to **Production** if:
1. No Production model exists yet (first deploy), or
2. Its `f1_churn` is strictly greater than the current Production model's `f1_churn`

This prevents a regression from silently overwriting a good production model.

---

## Other components (descriptions)

### Feature Store

In production I would use **Feast** with a Redis online store (for <5ms
inference latency) and Azure Blob offline store (for training) or Vertex AI Feature Store (if on GCP). The critical
requirement is point-in-time correct joins for the offline store — computing
a feature using data that was not available at the time of the label would
be target leakage.

The `FittedFeatures` dataclass in this implementation demonstrates the same
principle at small scale: training statistics are computed once and frozen,
then reused verbatim at inference time.

```python
# What the Feast version would look like:
feature_store = FeatureStore(repo_path=".")
training_df = feature_store.get_historical_features(entity_df, feature_refs).to_df()
online_features = feature_store.get_online_features(feature_refs, entity_rows).to_df()
```

### Monitoring

I would track two things:

**Data drift** — Population Stability Index (PSI) on `MonthlyCharges`, `tenure`,
`TotalCharges` comparing the training distribution against a rolling window of
inference requests. Given the 26% churn base rate in this dataset, a PSI > 0.2
on `MonthlyCharges` would trigger a retraining alert.

**Model performance** — When ground truth becomes available (actual churn after
30 days), compute `f1_churn` on labeled inference data and compare to the
Production model's training-time score. A drop > 0.05 triggers retraining.

**Application observability** — `src/serve.py` logs inference latency per
request. In production, these logs feed into **Prometheus** (scrapes the
`/metrics` endpoint exposed by the FastAPI app) + **Grafana** dashboards
tracking p95 latency, request volume, and error rate. Alertmanager handles
PagerDuty routing for SLO breaches. Evidently generates weekly drift reports.

### Orchestration (Retraining pipeline)

```python
# Airflow DAG — runs every Monday 02:00 UTC
@dag(schedule="0 2 * * 1")
def telco_churn_retraining():
    validate = validate_data_quality()    # Great Expectations suite
    features = run_feature_pipeline()     # depends on validate
    train    = train_and_register()       # depends on features
    promote  = evaluate_and_promote()     # depends on train
    notify   = slack_notification()       # depends on promote, trigger_rule=all_done
```

The branch after `evaluate_and_promote` matters: if the new model does not
beat Production, the pipeline ends without touching Production — a failed
improvement is not a pipeline failure.

### Deployment (CI/CD)

The existing CI pipeline (flake8 + black + pytest + docker build) would be
extended with two additional stages:

1. **Build** — build `Dockerfile.serve` into a Docker image tagged with the
   git SHA, pushed to GHCR.
2. **Deploy** — `kubectl set image` on the inference Deployment in K8s.
   The container loads the model at startup:
   `mlflow.sklearn.load_model("models:/telco-churn/Production")`

The registry is the deployment contract — the same artifact promoted via
`make train-promote` is what `src/serve.py` loads in production.
Rollback is `kubectl rollout undo`, triggered automatically if the post-deploy
`/health` check fails.
