"""
API tests for Customer Churn Prediction API
"""
import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app
from src.models.predict import ChurnPredictor

# Create test client
client = TestClient(app)

# Manually load model for testing (since TestClient doesn't trigger startup events)
@pytest.fixture(scope="session", autouse=True)
def load_test_model():
    """Load model before running tests"""
    try:
        import api.main
        if not api.main.model_loaded:
            # Manually load the model
            api.main.predictor = ChurnPredictor(
                model_path="models/best_model.joblib",
                artifacts_path="models/feature_artifacts"
            )
            api.main.model_loaded = True
    except Exception as e:
        pytest.fail(f"Failed to load model for testing: {e}")


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check(self):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "api_version" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestPredictionEndpoint:
    """Tests for prediction endpoints"""
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for testing"""
        return {
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
        }
    
    def test_single_prediction(self, sample_customer):
        """Test single customer prediction"""
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "churn_label" in data
        assert "risk_level" in data
        
        # Check data types
        assert isinstance(data["churn_prediction"], int)
        assert isinstance(data["churn_probability"], float)
        assert data["churn_prediction"] in [0, 1]
        assert 0 <= data["churn_probability"] <= 1
    
    def test_single_prediction_low_risk_customer(self):
        """Test prediction for low-risk customer"""
        low_risk_customer = {
            "customerID": "LOW_RISK",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 60,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 85.50,
            "TotalCharges": 5130.0
        }
        
        response = client.post("/predict", json=low_risk_customer)
        assert response.status_code == 200
        
        data = response.json()
        # Long tenure + two year contract = likely low churn probability
        assert data["churn_probability"] < 0.5
    
    def test_batch_prediction(self, sample_customer):
        """Test batch prediction"""
        batch_data = {
            "customers": [sample_customer, sample_customer]
        }
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_customers" in data
        assert "churn_count" in data
        assert "no_churn_count" in data
        assert "average_churn_probability" in data
        
        assert len(data["predictions"]) == 2
        assert data["total_customers"] == 2
    
    def test_invalid_input(self):
        """Test prediction with invalid input"""
        invalid_data = {
            "gender": "Invalid",
            "SeniorCitizen": 5  # Invalid value
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error


class TestModelInfoEndpoint:
    """Tests for model info endpoint"""
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_type" in data
        assert "n_features" in data
        assert "feature_names" in data
    
    def test_features_endpoint(self):
        """Test features endpoint"""
        response = client.get("/features")
        assert response.status_code == 200
        
        data = response.json()
        assert "categorical_features" in data
        assert "numerical_features" in data
        assert "total_features" in data


def run_tests():
    """Run all tests"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()