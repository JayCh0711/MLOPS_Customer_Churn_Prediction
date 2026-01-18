📁 Step 1: Project Setup
Bash

# Create project directory
mkdir mlops-customer-churn
cd mlops-customer-churn

# Open in VS Code
code .

# Create directory structure
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/utils
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p notebooks
mkdir -p tests
mkdir -p api
mkdir -p config
mkdir -p scripts
mkdir -p monitoring
mkdir -p .github/workflows

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/utils/__init__.py
touch api/__init__.py
touch tests/__init__.py

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch models/.gitkeep

# Initialize Git repository
git init
git add .
git commit -m "Initial project structure setup"
📊 Step 2: Data & DVC Setup
Bash

# Download dataset
python scripts/download_data.py

# Initialize DVC
dvc init

# Add DVC remote (local for testing)
dvc remote add -d localremote /tmp/dvc-storage

# Add DVC remote (Google Drive)
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID

# Track data with DVC
dvc add data/raw/telco_churn.csv

# View DVC file
cat data/raw/telco_churn.csv.dvc

# Test data loading
python src/data/load_data.py

# Test config loading
python src/utils/config.py

# Commit DVC files to Git
git add data/raw/telco_churn.csv.dvc
git add data/raw/.gitignore
git add .dvc/
git add .dvcignore
git add dvc.yaml
git add src/
git add config/
git add scripts/
git commit -m "Add dataset with DVC tracking and data loading utilities"

# Push data to DVC remote
dvc push

# Pull data from DVC remote (when needed)
dvc pull

# Check DVC status
dvc status
🔧 Step 3: Data Preprocessing & Feature Engineering
Bash

# Run preprocessing
python src/data/preprocess.py

# Run feature engineering
python src/features/build_features.py

# Run data splitting
python src/data/split_data.py

# Run complete DVC pipeline
dvc repro

# Run specific stage
dvc repro preprocess
dvc repro featurize
dvc repro split

# View pipeline DAG
dvc dag

# Track processed data with DVC
dvc add data/processed/cleaned_data.csv
dvc add data/processed/featured_data.csv
dvc add data/processed/X_train.csv
dvc add data/processed/X_test.csv
dvc add data/processed/y_train.csv
dvc add data/processed/y_test.csv

# Commit changes
git add src/data/preprocess.py
git add src/data/split_data.py
git add src/features/build_features.py
git add dvc.yaml
git add params.yaml
git add dvc.lock
git add data/processed/*.dvc
git add data/processed/.gitignore
git commit -m "Add data preprocessing and feature engineering pipeline"

# Push data to DVC
dvc push
🤖 Step 4: Model Training with MLflow
Bash

# Run model training
python src/models/train.py

# Run model evaluation
python src/models/evaluate.py

# Run prediction test
python src/models/predict.py

# Start MLflow UI
mlflow ui --port 5000

# Access MLflow UI in browser
# http://localhost:5000

# View DVC metrics
dvc metrics show

# Compare parameters
dvc params diff

# Run complete pipeline with training
dvc repro

# Commit changes
git add src/models/train.py
git add src/models/evaluate.py
git add src/models/predict.py
git add dvc.yaml
git add dvc.lock
git add models/metrics/*.json
git add models/evaluation/*.png
git commit -m "Add model training with MLflow tracking and evaluation"

# Push data to DVC
dvc push
🌐 Step 5: FastAPI REST API
Bash

# Run FastAPI server
python run_api.py

# Or run with uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation
# Swagger UI: http://localhost:8000/docs
# ReDoc: http://localhost:8000/redoc

# Test health endpoint
curl http://localhost:8000/health

# Test single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "TEST001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 1069.20
  }'

# Test model info endpoint
curl http://localhost:8000/model/info

# Run API tests
pytest tests/test_api.py -v

# Run API client demo
python api/client.py

# Commit changes
git add api/
git add run_api.py
git add tests/test_api.py
git commit -m "Add FastAPI REST API for model serving"
🐳 Step 6: Docker Containerization
Bash

# Build Docker image
docker build -t churn-prediction-api:latest .

# View built images
docker images | grep churn-prediction

# Run container manually
docker run -d \
    --name churn-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models:ro \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/config:/app/config:ro \
    churn-prediction-api:latest

# Check running containers
docker ps

# View container logs
docker logs churn-api

# Stop container
docker stop churn-api

# Remove container
docker rm churn-api

# Docker Compose commands
# Start API service
docker-compose up -d api

# Start all services
docker-compose up -d

# View running services
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild and start
docker-compose up -d --build api

# Open shell in container
docker-compose exec api /bin/bash

# Run production config
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Test Docker API
python scripts/test_docker_api.py

# Using Makefile commands
make help
make docker-build
make docker-up
make docker-logs
make docker-down
make docker-test
make clean
make clean-docker

# Commit Docker files
git add Dockerfile
git add Dockerfile.training
git add .dockerignore
git add docker-compose.yml
git add docker-compose.override.yml
git add docker-compose.prod.yml
git add Makefile
git add scripts/docker_start.sh
git add scripts/test_docker_api.py
git add nginx/
git commit -m "Add Docker containerization with docker-compose"
🔄 Step 7: CI/CD with GitHub Actions
Bash

# Run tests locally before pushing
pytest tests/ -v --tb=short

# Run linting
pip install flake8 black isort
black --check --diff src/ api/ tests/
isort --check-only --diff src/ api/ tests/
flake8 src/ api/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run security checks
pip install safety bandit
safety check --full-report
bandit -r src/ api/ -ll -ii

# Format code
black src/ api/ tests/
isort src/ api/ tests/

# Run model tests
pytest tests/test_model.py -v

# Create GitHub repository (using GitHub CLI)
gh repo create mlops-customer-churn --public

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/mlops-customer-churn.git
git push -u origin main

# Create a release tag
git tag -a v1.0.0 -m "First release"
git push origin v1.0.0

# View GitHub Actions status
gh run list
gh run view

# Commit CI/CD files
git add .github/
git add .flake8
git add pyproject.toml
git add tests/test_model.py
git commit -m "Add CI/CD pipeline with GitHub Actions"
git push
☁️ Step 8: Azure Cloud Deployment
Bash

# Install Azure CLI
# Windows:
winget install Microsoft.AzureCLI
# macOS:
brew install azure-cli
# Ubuntu/Debian:
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Verify Azure CLI
az --version

# Login to Azure
az login

# List subscriptions
az account list --output table

# Set subscription
az account set --subscription "YOUR_SUBSCRIPTION_NAME"

# Verify current subscription
az account show

# Create Azure resources
./scripts/azure/setup_azure.sh

# Create Service Principal for GitHub Actions
./scripts/azure/create_service_principal.sh

# Configure DVC with Azure Blob Storage
pip install dvc-azure
./scripts/azure/configure_dvc_azure.sh

# Push data to Azure DVC remote
dvc push

# Deploy infrastructure with Bicep
./scripts/azure/deploy_infrastructure.sh prod

# Deploy to Azure Container Apps
./scripts/azure/deploy_container_app.sh latest

# Login to Azure Container Registry
az acr login --name YOUR_ACR_NAME

# Build and push to ACR
docker build -t YOUR_ACR.azurecr.io/churn-prediction-api:latest .
docker push YOUR_ACR.azurecr.io/churn-prediction-api:latest

# View Container App logs
az containerapp logs show \
    --name ca-churn-api \
    --resource-group rg-mlops-churn

# Get Container App URL
az containerapp show \
    --name ca-churn-api \
    --resource-group rg-mlops-churn \
    --query "properties.configuration.ingress.fqdn" -o tsv

# Delete Azure resources (cleanup)
az group delete --name rg-mlops-churn --yes --no-wait

# Commit Azure configuration
git add scripts/azure/
git add infrastructure/
git add .github/workflows/azure-deploy.yml
git commit -m "Add Azure deployment configuration and infrastructure"
📈 Step 9: Monitoring & Drift Detection
Bash

# Create monitoring directories
mkdir -p monitoring/drift
mkdir -p monitoring/reports
mkdir -p monitoring/profiles
mkdir -p monitoring/alerts

# Run drift analysis
python monitoring/drift_detector.py

# Run data profiling
python monitoring/data_profiler.py

# Run monitoring service demo
python monitoring/monitoring_service.py

# Run alerting system demo
python monitoring/alerts.py

# Generate monitoring dashboard
python monitoring/dashboard_data.py

# Run complete monitoring demo
python scripts/run_monitoring_demo.py

# Run scheduled monitoring (runs continuously)
python monitoring/scheduled_monitoring.py

# Open dashboard in browser
# Open monitoring/dashboard.html

# Test monitoring API endpoints
curl http://localhost:8000/monitoring/metrics
curl http://localhost:8000/monitoring/alerts
curl http://localhost:8000/monitoring/status
curl -X POST http://localhost:8000/monitoring/drift-check
curl http://localhost:8000/monitoring/report

# Commit monitoring code
git add monitoring/
git add api/monitoring_routes.py
git add scripts/run_monitoring_demo.py
git commit -m "Add comprehensive monitoring system with drift detection and alerting"
🔧 General Utility Commands
Bash

# ==========================================
# Virtual Environment
# ==========================================
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
deactivate

# ==========================================
# Package Management
# ==========================================
pip install -r requirements.txt
pip freeze > requirements.txt
pip list
pip show <package_name>

# ==========================================
# Git Commands
# ==========================================
git status
git add .
git add <file>
git commit -m "message"
git push
git pull
git log --oneline
git branch
git checkout -b <branch_name>
git merge <branch_name>
git stash
git stash pop

# ==========================================
# DVC Commands
# ==========================================
dvc init
dvc add <file>
dvc push
dvc pull
dvc repro
dvc status
dvc dag
dvc metrics show
dvc params diff
dvc remote add -d <name> <path>
dvc remote list

# ==========================================
# MLflow Commands
# ==========================================
mlflow ui --port 5000
mlflow experiments list
mlflow runs list --experiment-id <id>

# ==========================================
# Docker Commands
# ==========================================
docker build -t <image_name> .
docker run -d -p 8000:8000 <image_name>
docker ps
docker ps -a
docker logs <container_id>
docker stop <container_id>
docker rm <container_id>
docker images
docker rmi <image_id>
docker exec -it <container_id> /bin/bash
docker-compose up -d
docker-compose down
docker-compose logs -f
docker-compose ps
docker system prune

# ==========================================
# Testing Commands
# ==========================================
pytest
pytest -v
pytest tests/test_api.py -v
pytest --cov=src --cov=api
pytest -v --tb=short

# ==========================================
# Code Quality Commands
# ==========================================
black src/ api/ tests/
black --check --diff src/ api/ tests/
isort src/ api/ tests/
isort --check-only --diff src/ api/ tests/
flake8 src/ api/ tests/
bandit -r src/ api/
safety check

# ==========================================
# Azure CLI Commands
# ==========================================
az login
az account list
az account set --subscription "<name>"
az account show
az group create --name <rg_name> --location <location>
az group delete --name <rg_name> --yes
az acr login --name <acr_name>
az containerapp logs show --name <app_name> --resource-group <rg_name>

# ==========================================
# Cleanup Commands
# ==========================================
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} +
rm -rf mlruns/
rm -rf .coverage htmlcov/
📋 Quick Reference: Common Workflows
Start Development
Bash

cd mlops-customer-churn
source venv/bin/activate
git pull
dvc pull
Run Full Pipeline
Bash

dvc repro
Start Local API
Bash

python run_api.py
# Or with Docker
docker-compose up -d api
Run Tests
Bash

pytest tests/ -v
Deploy to Production
Bash

git add .
git commit -m "Your message"
git push
# GitHub Actions will handle CI/CD
# For manual release:
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin v1.0.1
Run Monitoring
Bash

python scripts/run_monitoring_demo.py
# Open monitoring/dashboard.html in browser
📁 Project File Structure Summary
text

mlops-customer-churn/
├── .dvc/                    # DVC configuration
├── .github/workflows/       # GitHub Actions CI/CD
├── api/                     # FastAPI application
├── config/                  # Configuration files
├── data/
│   ├── raw/                # Raw data (DVC tracked)
│   └── processed/          # Processed data (DVC tracked)
├── infrastructure/          # Azure Bicep templates
├── models/                  # Trained models and artifacts
├── monitoring/              # Monitoring and drift detection
├── nginx/                   # Nginx configuration
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── src/
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model training
│   └── utils/             # Utilities
├── tests/                  # Test files
├── .dockerignore
├── .flake8
├── .gitignore
├── docker-compose.yml
├── docker-compose.override.yml
├── docker-compose.prod.yml
├── Dockerfile
├── dvc.lock
├── dvc.yaml
├── Makefile
├── params.yaml
├── pyproject.toml
├── requirements.txt
└── run_api.py