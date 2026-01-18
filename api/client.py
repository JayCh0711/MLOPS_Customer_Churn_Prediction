"""
Python client for Customer Churn Prediction API
"""
import requests
from typing import Dict, List, Optional
import json


class ChurnAPIClient:
    """
    Client for interacting with Churn Prediction API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client
        
        Args:
            base_url: API base URL
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """Check API health"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Health check failed: {e}")
            return {"error": "Server not accessible", "status": "unhealthy"}
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Model info request failed: {e}")
            return {"error": "Request failed"}
    
    def predict(self, customer_data: Dict) -> Dict:
        """
        Make single prediction
        
        Args:
            customer_data: Customer data dictionary
            
        Returns:
            Prediction result
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json=customer_data
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, customers: List[Dict]) -> Dict:
        """
        Make batch prediction
        
        Args:
            customers: List of customer data dictionaries
            
        Returns:
            Batch prediction results
        """
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"customers": customers}
        )
        response.raise_for_status()
        return response.json()


def demo_client():
    """Demonstrate API client usage"""
    
    # Initialize client
    client = ChurnAPIClient("http://localhost:8000")
    
    print("="*60)
    print("CHURN PREDICTION API CLIENT DEMO")
    print("="*60)
    
    # 1. Health check
    print("\n1. Health Check:")
    health = client.health_check()
    
    if "error" in health:
        print("❌ API server is not running!")
        print("Please start the server first with:")
        print("   $env:DISABLE_RELOAD = 'true'; python -m api.run_api")
        return
    
    print(f"   Status: {health['status']}")
    print(f"   Model Loaded: {health['model_loaded']}")
    
    # 2. Model info
    print("\n2. Model Info:")
    info = client.get_model_info()
    print(f"   Model Type: {info['model_type']}")
    print(f"   Number of Features: {info['n_features']}")
    
    # 3. Single prediction - High risk customer
    print("\n3. Single Prediction (High Risk Customer):")
    high_risk = {
        "customerID": "HIGH_RISK_001",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
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
        "MonthlyCharges": 95.00,
        "TotalCharges": 95.00
    }
    
    result = client.predict(high_risk)
    print(f"   Customer ID: {result['customerID']}")
    print(f"   Prediction: {result['churn_label']}")
    print(f"   Probability: {result['churn_probability']:.2%}")
    print(f"   Risk Level: {result['risk_level']}")
    
    # 4. Single prediction - Low risk customer
    print("\n4. Single Prediction (Low Risk Customer):")
    low_risk = {
        "customerID": "LOW_RISK_001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.00,
        "TotalCharges": 4680.00
    }
    
    result = client.predict(low_risk)
    print(f"   Customer ID: {result['customerID']}")
    print(f"   Prediction: {result['churn_label']}")
    print(f"   Probability: {result['churn_probability']:.2%}")
    print(f"   Risk Level: {result['risk_level']}")
    
    # 5. Batch prediction
    print("\n5. Batch Prediction:")
    batch_result = client.predict_batch([high_risk, low_risk])
    print(f"   Total Customers: {batch_result['total_customers']}")
    print(f"   Churn Count: {batch_result['churn_count']}")
    print(f"   No Churn Count: {batch_result['no_churn_count']}")
    print(f"   Avg Churn Probability: {batch_result['average_churn_probability']:.2%}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)


if __name__ == "__main__":
    demo_client()