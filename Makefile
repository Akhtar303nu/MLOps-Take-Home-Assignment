.PHONY: help install install-dev lint test stack-up stack-down train train-promote dev

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install dev dependencies
	pip install -r requirements-dev.txt

lint: ## Run flake8 + black check
	flake8 src/ tests/
	black --check src/ tests/

test: ## Run tests with coverage
	PYTHONPATH=. pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

stack-up: ## Start MLflow stack (postgres + minio + mlflow)
	docker compose up -d
	@echo ""
	@echo "MLflow UI:     http://localhost:5000"
	@echo "MinIO console: http://localhost:9001  (minioadmin/minioadmin)"

stack-down: ## Tear down stack and volumes
	docker compose down -v

dev: ## Start MLflow locally (SQLite + local artifacts, no Docker)
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts \
		--host 127.0.0.1 \
		--port 5000 &

train: ## Train and register model — promotes to Staging if gates pass
	PYTHONPATH=. python3 -m src.train --data data/WA_Fn-UseC_-Telco-Customer-C.csv

train-promote: ## Train and promote to Production if model beats current Production
	PYTHONPATH=. python3 -m src.train --data data/WA_Fn-UseC_-Telco-Customer-C.csv --promote-to-production