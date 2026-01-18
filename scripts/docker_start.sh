#!/bin/bash

# ============================================
# Docker Startup Script
# ============================================

set -e

echo "============================================"
echo "Customer Churn Prediction - Docker Setup"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_status "Docker is running."

# Check if model exists
if [ ! -f "models/best_model.joblib" ]; then
    print_warning "Model not found. Running training pipeline first..."
    
    # Check if data exists
    if [ ! -f "data/raw/telco_churn.csv" ]; then
        print_status "Downloading data..."
        python scripts/download_data.py
    fi
    
    print_status "Running DVC pipeline..."
    dvc repro
fi

print_status "Model found. Starting containers..."

# Parse command line arguments
MODE=${1:-"production"}

case $MODE in
    "production")
        print_status "Starting in PRODUCTION mode..."
        docker-compose up -d api
        ;;
    "development")
        print_status "Starting in DEVELOPMENT mode..."
        docker-compose up api
        ;;
    "all")
        print_status "Starting ALL services..."
        docker-compose up -d
        ;;
    "training")
        print_status "Starting TRAINING service..."
        docker-compose --profile training up trainer
        ;;
    *)
        print_error "Unknown mode: $MODE"
        echo "Usage: $0 [production|development|all|training]"
        exit 1
        ;;
esac

print_status "Containers started!"
echo ""
echo "============================================"
echo "Services:"
echo "  - API:    http://localhost:8000"
echo "  - Docs:   http://localhost:8000/docs"
echo "  - MLflow: http://localhost:5000"
echo "============================================"