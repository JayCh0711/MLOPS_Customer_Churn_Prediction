"""
FastAPI application with Azure Application Insights
"""
import os
import sys
import logging
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import json

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas import (
    CustomerData, BatchCustomerData,
    PredictionResult, BatchPredictionResult,
    HealthResponse, ModelInfoResponse
)
from src.models.predict import ChurnPredictor

# ============================================
# Azure Application Insights Setup (Optional)
# ============================================
APPINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

if APPINSIGHTS_CONNECTION_STRING:
    try:
        from opencensus.ext.azure.log_exporter import AzureLogHandler
        from opencensus.ext.azure.trace_exporter import AzureExporter
        from opencensus.trace.samplers import ProbabilitySampler
        from opencensus.trace.tracer import Tracer
        
        # Configure logging with Azure
        logger = logging.getLogger(__name__)
        logger.addHandler(AzureLogHandler(connection_string=APPINSIGHTS_CONNECTION_STRING))
        
        # Configure tracing
        tracer = Tracer(
            exporter=AzureExporter(connection_string=APPINSIGHTS_CONNECTION_STRING),
            sampler=ProbabilitySampler(1.0)
        )
        
        AZURE_MONITORING_ENABLED = True
        logger.info("Azure Application Insights enabled")
    except ImportError:
        AZURE_MONITORING_ENABLED = False
        logger = logging.getLogger(__name__)
        logger.warning("Azure monitoring packages not installed")
else:
    AZURE_MONITORING_ENABLED = False
    logger = logging.getLogger(__name__)

# Configure basic logging
logging.basicConfig(level=logging.INFO)

# API metadata
API_TITLE = "Customer Churn Prediction API"
API_DESCRIPTION = """
## Customer Churn Prediction API

Production-ready API for predicting customer churn.

### Endpoints:
* **Single Prediction**: Predict churn for a single customer
* **Batch Prediction**: Predict churn for multiple customers
* **Model Info**: Get information about the deployed model
* **Health Check**: Check API and model health
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

# Global variables
predictor = None
model_loaded = False
startup_time = None


# ============================================
# Middleware for Request Logging
# ============================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    logger.info(
        f"{request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Duration: {process_time:.3f}s"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# ============================================
# Helper Functions
# ============================================
def get_risk_level(probability: float) -> str:
    """Get risk level based on churn probability"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


# ============================================
# Startup Event
# ============================================
@app.on_event("startup")
async def load_model():
    """Load model on application startup"""
    global predictor, model_loaded, startup_time
    
    startup_time = datetime.utcnow().isoformat()
    
    try:
        predictor = ChurnPredictor(
            model_path="models/best_model.joblib",
            artifacts_path="models/feature_artifacts"
        )
        model_loaded = True
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model_loaded = False


# ============================================
# Endpoints
# ============================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Customer Churn Prediction API",
        "version": API_VERSION,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        api_version=API_VERSION
    )


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Kubernetes readiness probe"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "ready"}


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """Get model information"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
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
    """Single customer churn prediction"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        customer_dict = customer.model_dump()
        result = predictor.predict(customer_dict)
        
        probability = result['churn_probability'][0]
        prediction = result['predictions'][0]
        
        logger.info(
            f"Prediction made - Customer: {customer.customerID}, "
            f"Churn: {prediction}, Probability: {probability:.4f}"
        )
        
        return PredictionResult(
            customerID=customer.customerID,
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            churn_label="Churn" if prediction == 1 else "No Churn",
            risk_level=get_risk_level(probability)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResult, tags=["Predictions"])
async def predict_batch(batch: BatchCustomerData):
    """Batch customer churn prediction"""
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        customers_list = [c.model_dump() for c in batch.customers]
        result = predictor.predict(customers_list)
        
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
        
        churn_count = sum(result['predictions'])
        no_churn_count = len(result['predictions']) - churn_count
        avg_probability = sum(result['churn_probability']) / len(result['churn_probability'])
        
        logger.info(f"Batch prediction - {len(predictions)} customers processed")
        
        return BatchPredictionResult(
            predictions=predictions,
            total_customers=len(predictions),
            churn_count=churn_count,
            no_churn_count=no_churn_count,
            average_churn_probability=round(avg_probability, 4)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/features", tags=["Model"])
async def get_feature_info():
    """Get feature information"""
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


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get API metrics for monitoring"""
    return {
        "startup_time": startup_time,
        "model_loaded": model_loaded,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": API_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


"""
Monitoring API Routes
"""
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitoring.monitoring_service import get_monitoring_service
from monitoring.drift_detector import DataDriftDetector
from monitoring.data_profiler import DataProfiler

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


# Response Models
class MetricsResponse(BaseModel):
    total_predictions: int
    churn_rate: Optional[float] = None
    average_probability: float
    high_risk_count: int
    low_risk_count: int
    high_risk_rate: Optional[float] = None


class AlertResponse(BaseModel):
    timestamp: str
    type: str
    message: str


class HealthStatusResponse(BaseModel):
    status: str
    total_predictions: int
    recent_alerts_count: int
    last_drift_check: Optional[str] = None


class DriftReportResponse(BaseModel):
    timestamp: str
    dataset_drift_detected: bool
    drift_share: float
    number_of_drifted_columns: int
    number_of_columns: int
    drifted_columns: List[str]


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get current monitoring metrics
    """
    try:
        service = get_monitoring_service()
        metrics = service.get_metrics()
        
        return MetricsResponse(
            total_predictions=metrics['total_predictions'],
            churn_rate=metrics.get('churn_rate'),
            average_probability=metrics['average_probability'],
            high_risk_count=metrics['high_risk_count'],
            low_risk_count=metrics['low_risk_count'],
            high_risk_rate=metrics.get('high_risk_rate')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[AlertResponse])
async def get_alerts(
    hours: int = Query(24, description="Get alerts from last N hours")
):
    """
    Get recent alerts
    """
    try:
        service = get_monitoring_service()
        since = datetime.now() - timedelta(hours=hours)
        alerts = service.get_alerts(since=since)
        
        return [
            AlertResponse(
                timestamp=a['timestamp'],
                type=a['type'],
                message=a['message']
            )
            for a in alerts
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=HealthStatusResponse)
async def get_monitoring_status():
    """
    Get overall monitoring status
    """
    try:
        service = get_monitoring_service()
        metrics = service.get_metrics()
        recent_alerts = service.get_alerts(
            since=datetime.now() - timedelta(hours=24)
        )
        
        # Determine status
        status = "healthy"
        if len(recent_alerts) > 0:
            status = "warning"
        if any(a['type'] in ['DATA_DRIFT', 'PERFORMANCE_DEGRADATION'] for a in recent_alerts):
            status = "critical"
        
        return HealthStatusResponse(
            status=status,
            total_predictions=metrics['total_predictions'],
            recent_alerts_count=len(recent_alerts),
            last_drift_check=None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/drift-check", response_model=DriftReportResponse)
async def run_drift_check():
    """
    Run drift detection on recent predictions
    """
    try:
        service = get_monitoring_service()
        result = service.run_drift_check()
        
        if result.get('status') == 'insufficient_data':
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for drift check. Need at least 50 samples, have {result.get('samples', 0)}"
            )
        
        return DriftReportResponse(
            timestamp=result['timestamp'],
            dataset_drift_detected=result['dataset_drift_detected'],
            drift_share=result['drift_share'],
            number_of_drifted_columns=result['number_of_drifted_columns'],
            number_of_columns=result['number_of_columns'],
            drifted_columns=result['drifted_columns']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/report")
async def generate_report():
    """
    Generate comprehensive monitoring report
    """
    try:
        service = get_monitoring_service()
        report = service.generate_report()
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))