"""
Test script for Docker containerized API
"""
import requests
import time
import sys


def wait_for_api(url: str, max_retries: int = 30, delay: int = 2) -> bool:
    """Wait for API to become available"""
    print(f"Waiting for API at {url}...")
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ API is ready! (attempt {i+1})")
                return True
        except requests.exceptions.ConnectionError:
            pass
        
        print(f"   Attempt {i+1}/{max_retries} - API not ready, waiting {delay}s...")
        time.sleep(delay)
    
    print("❌ API failed to start")
    return False


def test_api(base_url: str):
    """Run API tests"""
    print("\n" + "="*60)
    print("DOCKER API TESTS")
    print("="*60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health Check
    print("\n[Test 1] Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] == True
        print("   ✅ PASSED - API is healthy and model is loaded")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED - {e}")
        tests_failed += 1
    
    # Test 2: Model Info
    print("\n[Test 2] Model Info...")
    try:
        response = requests.get(f"{base_url}/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert data["n_features"] > 0
        print(f"   ✅ PASSED - Model type: {data['model_type']}, Features: {data['n_features']}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED - {e}")
        tests_failed += 1
    
    # Test 3: Single Prediction
    print("\n[Test 3] Single Prediction...")
    try:
        customer = {
            "customerID": "DOCKER_TEST",
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
        
        response = requests.post(f"{base_url}/predict", json=customer)
        assert response.status_code == 200
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        print(f"   ✅ PASSED - Prediction: {data['churn_label']}, Probability: {data['churn_probability']:.2%}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED - {e}")
        tests_failed += 1
    
    # Test 4: Batch Prediction
    print("\n[Test 4] Batch Prediction...")
    try:
        batch = {"customers": [customer, customer]}
        response = requests.post(f"{base_url}/predict/batch", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["total_customers"] == 2
        print(f"   ✅ PASSED - Processed {data['total_customers']} customers")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED - {e}")
        tests_failed += 1
    
    # Test 5: Invalid Input
    print("\n[Test 5] Invalid Input Handling...")
    try:
        invalid = {"gender": "Invalid"}
        response = requests.post(f"{base_url}/predict", json=invalid)
        assert response.status_code == 422  # Validation error
        print("   ✅ PASSED - Invalid input properly rejected")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED - {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"   ✅ Passed: {tests_passed}")
    print(f"   ❌ Failed: {tests_failed}")
    print(f"   Total:   {tests_passed + tests_failed}")
    print("="*60)
    
    return tests_failed == 0


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    # Wait for API
    if not wait_for_api(base_url):
        sys.exit(1)
    
    # Run tests
    success = test_api(base_url)
    sys.exit(0 if success else 1)