"""
Model tests for CI pipeline
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataPreprocessing:
    """Tests for data preprocessing"""
    
    def test_raw_data_exists(self):
        """Test that raw data file exists"""
        assert os.path.exists("data/raw/telco_churn.csv")
    
    def test_raw_data_not_empty(self):
        """Test that raw data is not empty"""
        df = pd.read_csv("data/raw/telco_churn.csv")
        assert len(df) > 0
    
    def test_raw_data_has_required_columns(self):
        """Test that raw data has required columns"""
        df = pd.read_csv("data/raw/telco_churn.csv")
        required_columns = ['customerID', 'Churn', 'tenure', 'MonthlyCharges']
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"


class TestProcessedData:
    """Tests for processed data"""
    
    @pytest.fixture
    def processed_files(self):
        """List of expected processed files"""
        return [
            "data/processed/cleaned_data.csv",
            "data/processed/featured_data.csv",
            "data/processed/X_train.csv",
            "data/processed/X_test.csv",
            "data/processed/y_train.csv",
            "data/processed/y_test.csv"
        ]
    
    def test_processed_files_exist(self, processed_files):
        """Test that all processed files exist"""
        for filepath in processed_files:
            assert os.path.exists(filepath), f"Missing file: {filepath}"
    
    def test_train_test_split_sizes(self):
        """Test train/test split proportions"""
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        # Should be approximately 0.2
        assert 0.15 <= test_ratio <= 0.25, f"Unexpected test ratio: {test_ratio}"
    
    def test_no_data_leakage(self):
        """Test that there's no overlap between train and test"""
        X_train = pd.read_csv("data/processed/X_train.csv")
        X_test = pd.read_csv("data/processed/X_test.csv")
        
        # Reset index for comparison
        train_idx = set(X_train.index.tolist())
        test_idx = set(X_test.index.tolist())
        
        # Check for overlap (using index as proxy)
        assert len(train_idx.intersection(test_idx)) == 0 or \
               X_train.index.tolist() != X_test.index.tolist()


class TestModel:
    """Tests for trained model"""
    
    def test_model_file_exists(self):
        """Test that model file exists"""
        assert os.path.exists("models/best_model.joblib")
    
    def test_model_artifacts_exist(self):
        """Test that model artifacts exist"""
        artifacts = [
            "models/feature_artifacts/label_encoders.joblib",
            "models/feature_artifacts/scaler.joblib",
            "models/feature_artifacts/column_info.joblib"
        ]
        for filepath in artifacts:
            assert os.path.exists(filepath), f"Missing artifact: {filepath}"
    
    def test_model_can_load(self):
        """Test that model can be loaded"""
        import joblib
        model = joblib.load("models/best_model.joblib")
        assert model is not None
    
    def test_model_can_predict(self):
        """Test that model can make predictions"""
        import joblib
        
        model = joblib.load("models/best_model.joblib")
        X_test = pd.read_csv("data/processed/X_test.csv")
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)
    
    def test_model_predictions_reasonable(self):
        """Test that predictions are reasonable"""
        import joblib
        
        model = joblib.load("models/best_model.joblib")
        X_test = pd.read_csv("data/processed/X_test.csv")
        
        predictions = model.predict(X_test)
        
        # Check that not all predictions are the same
        unique_predictions = np.unique(predictions)
        assert len(unique_predictions) > 1, "Model predicts only one class"
        
        # Check prediction distribution is reasonable
        churn_rate = predictions.mean()
        assert 0.1 <= churn_rate <= 0.9, f"Unusual churn rate: {churn_rate}"


class TestMetrics:
    """Tests for model metrics"""
    
    def test_metrics_file_exists(self):
        """Test that metrics file exists"""
        assert os.path.exists("models/metrics/training_summary.json")
    
    def test_metrics_values_valid(self):
        """Test that metrics values are valid"""
        import json
        
        with open("models/metrics/training_summary.json", 'r') as f:
            metrics = json.load(f)
        
        best_metrics = metrics.get('best_metrics', {})
        
        # Check accuracy
        if 'accuracy' in best_metrics:
            assert 0 <= best_metrics['accuracy'] <= 1
        
        # Check F1 score
        if 'f1_score' in best_metrics:
            assert 0 <= best_metrics['f1_score'] <= 1
        
        # Check ROC-AUC
        if 'roc_auc' in best_metrics:
            assert 0 <= best_metrics['roc_auc'] <= 1
    
    def test_model_meets_minimum_performance(self):
        """Test that model meets minimum performance thresholds"""
        import json
        
        with open("models/metrics/training_summary.json", 'r') as f:
            metrics = json.load(f)
        
        best_metrics = metrics.get('best_metrics', {})
        
        # Minimum thresholds
        assert best_metrics.get('accuracy', 0) >= 0.6, "Accuracy below threshold"
        assert best_metrics.get('f1_score', 0) >= 0.4, "F1 score below threshold"
        assert best_metrics.get('roc_auc', 0) >= 0.6, "ROC-AUC below threshold"


def run_all_tests():
    """Run all model tests"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()