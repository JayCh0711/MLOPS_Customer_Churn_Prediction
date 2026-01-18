"""
Model prediction module
"""
import pandas as pd
import numpy as np
import joblib
import logging
import os
from typing import Union, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Churn prediction class for making predictions on new data
    """
    
    def __init__(
        self,
        model_path: str = "models/best_model.joblib",
        artifacts_path: str = "models/feature_artifacts"
    ):
        """
        Initialize predictor with model and feature artifacts
        
        Args:
            model_path: Path to trained model
            artifacts_path: Path to feature engineering artifacts
        """
        # Load model
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load feature artifacts
        self.label_encoders = joblib.load(
            os.path.join(artifacts_path, 'label_encoders.joblib')
        )
        self.scaler = joblib.load(
            os.path.join(artifacts_path, 'scaler.joblib')
        )
        self.column_info = joblib.load(
            os.path.join(artifacts_path, 'column_info.joblib')
        )
        
        logger.info(f"Loaded feature artifacts from {artifacts_path}")
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data for prediction
        
        Args:
            data: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        # Handle TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(0, inplace=True)
        
        # Convert SeniorCitizen
        if 'SeniorCitizen' in df.columns:
            if df['SeniorCitizen'].dtype in ['int64', 'float64']:
                df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # Remove customerID if present
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        # Remove Churn if present (for prediction)
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for prediction
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Create tenure group
        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 60, np.inf],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr']
        )
        
        # Create monthly charges group
        df['monthly_charges_group'] = pd.cut(
            df['MonthlyCharges'],
            bins=[0, 30, 60, 90, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Average monthly spend
        df['avg_monthly_spend'] = np.where(
            df['tenure'] > 0,
            df['TotalCharges'] / df['tenure'],
            df['MonthlyCharges']
        )
        
        # Number of services
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        def count_services(row):
            count = 0
            for col in service_cols:
                if col in row.index:
                    if row[col] not in ['No', 'No internet service', 'No phone service']:
                        count += 1
            return count
        
        df['num_services'] = df.apply(count_services, axis=1)
        
        # Has family
        df['has_family'] = np.where(
            (df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes'),
            'Yes', 'No'
        )
        
        # Contract risk
        contract_risk = {'Month-to-month': 3, 'One year': 2, 'Two year': 1}
        df['contract_risk'] = df['Contract'].map(contract_risk)
        
        # Charges per tenure
        df['charges_per_tenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        return df
    
    def encode_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical and scale numerical features
        
        Args:
            df: DataFrame with features
            
        Returns:
            Encoded and scaled DataFrame
        """
        categorical_cols = self.column_info['categorical_columns']
        numerical_cols = self.column_info['numerical_columns']
        
        # Encode categorical
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        # Scale numerical
        existing_num_cols = [c for c in numerical_cols if c in df.columns]
        if existing_num_cols:
            df[existing_num_cols] = self.scaler.transform(df[existing_num_cols])
        
        return df
    
    def predict(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> Dict:
        """
        Make prediction on input data
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        # Convert to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Preprocess
        df = self.preprocess_input(df)
        
        # Create features
        df = self.create_features(df)
        
        # Encode and scale
        df = self.encode_and_scale(df)
        
        # Ensure columns match training data
        expected_features = self.column_info['feature_names']
        
        # Add missing columns with 0
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0
        
        # Select only expected features in correct order
        df = df[expected_features]
        
        # Make predictions
        predictions = self.model.predict(df)
        probabilities = self.model.predict_proba(df)[:, 1]
        
        results = {
            'predictions': predictions.tolist(),
            'churn_probability': probabilities.tolist(),
            'churn_labels': ['Churn' if p == 1 else 'No Churn' for p in predictions]
        }
        
        return results


def predict_sample():
    """Test prediction with sample data"""
    # Sample customer data
    sample_data = {
        'customerID': 'TEST001',
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 89.10,
        'TotalCharges': 1069.2
    }
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Make prediction
    result = predictor.predict(sample_data)
    
    print("\n" + "="*50)
    print("SAMPLE PREDICTION")
    print("="*50)
    print(f"Customer: {sample_data['customerID']}")
    print(f"Prediction: {result['churn_labels'][0]}")
    print(f"Churn Probability: {result['churn_probability'][0]:.4f}")
    print("="*50)
    
    return result


if __name__ == "__main__":
    predict_sample()