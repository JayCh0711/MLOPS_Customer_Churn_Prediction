# ============================================
# Makefile for MLOps Customer Churn Project
# ============================================

.PHONY: help install train api docker-build docker-up docker-down docker-logs test clean

# Default target
help:
	@echo "============================================"
	@echo "MLOps Customer Churn - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install dependencies"
	@echo "  make download-data  - Download dataset"
	@echo ""
	@echo "Development:"
	@echo "  make train          - Run training pipeline"
	@echo "  make api            - Run API locally"
	@echo "  make test           - Run tests"
	@echo "  make mlflow         - Start MLflow UI"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start Docker containers"
	@echo "  make docker-down    - Stop Docker containers"
	@echo "  make docker-logs    - View Docker logs"
	@echo "  make docker-test    - Test Docker API"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Clean temporary files"
	@echo "  make clean-docker   - Remove Docker containers/images"
	@echo ""

# ============================================
# Setup Commands
# ============================================

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

download-data:
	@echo "Downloading dataset..."
	python scripts/download_data.py

# ============================================
# Development Commands
# ============================================

train:
	@echo "Running training pipeline..."
	dvc repro

api:
	@echo "Starting API server..."
	python run_api.py

test:
	@echo "Running tests..."
	pytest tests/ -v --tb=short

mlflow:
	@echo "Starting MLflow UI..."
	mlflow ui --port 5000

# ============================================
# Docker Commands
# ============================================

docker-build:
	@echo "Building Docker image..."
	docker build -t churn-prediction-api:latest .

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d api
	@echo "Waiting for API to start..."
	@sleep 5
	@echo "API available at http://localhost:8000"
	@echo "Docs available at http://localhost:8000/docs"

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down

docker-logs:
	@echo "Showing Docker logs..."
	docker-compose logs -f api

docker-test:
	@echo "Testing Docker API..."
	python scripts/test_docker_api.py

docker-all:
	@echo "Starting all services..."
	docker-compose up -d

docker-shell:
	@echo "Opening shell in API container..."
	docker-compose exec api /bin/bash

# ============================================
# Cleanup Commands
# ============================================

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ 2>/dev/null || true
	@echo "Done!"

clean-docker:
	@echo "Removing Docker containers and images..."
	docker-compose down --rmi local -v
	@echo "Done!"