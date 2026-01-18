"""
FastAPI application for Customer Churn Prediction
"""
import os
import sys
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import json

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    CustomerData, BatchCustomerData,
    PredictionResult, BatchPredictionResult,
    HealthResponse, ModelInfoResponse
)
from src.models.predict import ChurnPredictor

# API metadata
API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = """
## Customer Churn Prediction API

This API provides endpoints for predicting customer churn using a machine learning model.

### Features:
* **Single Prediction**: Predict churn for a single customer
* **Batch Prediction**: Predict churn for multiple customers
* **Model Info**: Get information about the deployed model
* **Health Check**: Check API and model health

### Model:
The model is trained on Telco Customer Churn dataset and uses ensemble methods
for prediction.
"""
API_VERSION = "1.0.0"

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
predictor = None
model_loaded = False


def get_risk_level(probability: float) -> str:
    """
    Get risk level based on churn probability
    
    Args:
        probability: Churn probability
        
    Returns:
        Risk level string
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


@app.on_event("startup")
async def load_model():
    """Load model on application startup"""
    global predictor, model_loaded
    
    try:
        # Get the project root directory (parent of api directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, "models", "best_model.joblib")
        artifacts_path = os.path.join(project_root, "models", "feature_artifacts")
        
        predictor = ChurnPredictor(
            model_path=model_path,
            artifacts_path=artifacts_path
        )
        model_loaded = True
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to Customer Churn Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Returns API and model health status
    """
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        api_version=API_VERSION
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get model information
    
    Returns details about the deployed model
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Load training summary if available
        metrics = {}
        metrics_path = "models/metrics/training_summary.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                summary = json.load(f)
                metrics = summary.get('best_metrics', {})
        
        return ModelInfoResponse(
            model_type=type(predictor.model).__name__,
            n_features=len(predictor.column_info['feature_names']),
            feature_names=predictor.column_info['feature_names'],
            training_date=None,
            metrics=metrics
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResult, tags=["Predictions"])
async def predict_single(customer: CustomerData):
    """
    Single customer churn prediction
    
    Predicts whether a single customer will churn
    
    - **customerID**: Optional customer identifier
    - **gender**: Customer gender (Male/Female)
    - **SeniorCitizen**: Is senior citizen (0 or 1)
    - **Partner**: Has partner (Yes/No)
    - ... (other features)
    
    Returns prediction, probability, and risk level
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert to dict
        customer_dict = customer.model_dump()
        
        # Make prediction
        result = predictor.predict(customer_dict)
        
        # Get probability
        probability = result['churn_probability'][0]
        prediction = result['predictions'][0]
        
        return PredictionResult(
            customerID=customer.customerID,
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            churn_label="Churn" if prediction == 1 else "No Churn",
            risk_level=get_risk_level(probability)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Predictions"])
async def predict_batch(batch: BatchCustomerData):
    """
    Batch customer churn prediction
    
    Predicts churn for multiple customers at once
    
    - **customers**: List of customer data objects
    
    Returns predictions for all customers with summary statistics
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Convert to list of dicts
        customers_list = [c.model_dump() for c in batch.customers]
        
        # Make predictions
        result = predictor.predict(customers_list)
        
        # Build response
        predictions = []
        for i, customer in enumerate(batch.customers):
            probability = result['churn_probability'][i]
            prediction = result['predictions'][i]
            
            predictions.append(PredictionResult(
                customerID=customer.customerID,
                churn_prediction=prediction,
                churn_probability=round(probability, 4),
                churn_label="Churn" if prediction == 1 else "No Churn",
                risk_level=get_risk_level(probability)
            ))
        
        # Calculate summary
        churn_count = sum(result['predictions'])
        no_churn_count = len(result['predictions']) - churn_count
        avg_probability = sum(result['churn_probability']) / len(result['churn_probability'])
        
        return BatchPredictionResult(
            predictions=predictions,
            total_customers=len(predictions),
            churn_count=churn_count,
            no_churn_count=no_churn_count,
            average_churn_probability=round(avg_probability, 4)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/features", tags=["Model"])
async def get_feature_info():
    """
    Get feature information
    
    Returns list of features expected by the model
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "categorical_features": predictor.column_info['categorical_columns'],
        "numerical_features": predictor.column_info['numerical_columns'],
        "total_features": len(predictor.column_info['feature_names'])
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An unexpected error occurred",
            "error": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )