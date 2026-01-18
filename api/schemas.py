"""
Pydantic schemas for API request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum


class GenderEnum(str, Enum):
    male = "Male"
    female = "Female"


class YesNoEnum(str, Enum):
    yes = "Yes"
    no = "No"


class InternetServiceEnum(str, Enum):
    dsl = "DSL"
    fiber_optic = "Fiber optic"
    no = "No"


class ContractEnum(str, Enum):
    month_to_month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethodEnum(str, Enum):
    electronic_check = "Electronic check"
    mailed_check = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class MultilineLinesEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone_service = "No phone service"


class InternetDependentEnum(str, Enum):
    yes = "Yes"
    no = "No"
    no_internet_service = "No internet service"


# Request Schema
class CustomerData(BaseModel):
    """
    Schema for single customer prediction request
    """
    customerID: Optional[str] = Field(None, description="Customer ID (optional)")
    gender: GenderEnum = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0 or 1)")
    Partner: YesNoEnum = Field(..., description="Has partner")
    Dependents: YesNoEnum = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: YesNoEnum = Field(..., description="Has phone service")
    MultipleLines: MultilineLinesEnum = Field(..., description="Has multiple lines")
    InternetService: InternetServiceEnum = Field(..., description="Type of internet service")
    OnlineSecurity: InternetDependentEnum = Field(..., description="Has online security")
    OnlineBackup: InternetDependentEnum = Field(..., description="Has online backup")
    DeviceProtection: InternetDependentEnum = Field(..., description="Has device protection")
    TechSupport: InternetDependentEnum = Field(..., description="Has tech support")
    StreamingTV: InternetDependentEnum = Field(..., description="Has streaming TV")
    StreamingMovies: InternetDependentEnum = Field(..., description="Has streaming movies")
    Contract: ContractEnum = Field(..., description="Contract type")
    PaperlessBilling: YesNoEnum = Field(..., description="Has paperless billing")
    PaymentMethod: PaymentMethodEnum = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges")
    TotalCharges: float = Field(..., ge=0, description="Total charges")

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "CUST001",
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
            }
        }


# Batch Request Schema
class BatchCustomerData(BaseModel):
    """
    Schema for batch prediction request
    """
    customers: List[CustomerData] = Field(..., description="List of customers")

    class Config:
        json_schema_extra = {
            "example": {
                "customers": [
                    {
                        "customerID": "CUST001",
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
                    }
                ]
            }
        }


# Response Schemas
class PredictionResult(BaseModel):
    """
    Schema for single prediction response
    """
    customerID: Optional[str] = Field(None, description="Customer ID")
    churn_prediction: int = Field(..., description="Prediction (0 or 1)")
    churn_probability: float = Field(..., description="Probability of churn")
    churn_label: str = Field(..., description="Human readable label")
    risk_level: str = Field(..., description="Risk level (Low, Medium, High)")

    class Config:
        json_schema_extra = {
            "example": {
                "customerID": "CUST001",
                "churn_prediction": 1,
                "churn_probability": 0.75,
                "churn_label": "Churn",
                "risk_level": "High"
            }
        }


class BatchPredictionResult(BaseModel):
    """
    Schema for batch prediction response
    """
    predictions: List[PredictionResult]
    total_customers: int
    churn_count: int
    no_churn_count: int
    average_churn_probability: float


class HealthResponse(BaseModel):
    """
    Schema for health check response
    """
    status: str
    model_loaded: bool
    api_version: str


class ModelInfoResponse(BaseModel):
    """
    Schema for model info response
    """
    model_type: str
    n_features: int
    feature_names: List[str]
    training_date: Optional[str]
    metrics: dict