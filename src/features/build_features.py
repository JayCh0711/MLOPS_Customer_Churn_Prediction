"""
Feature engineering module for Telco Customer Churn dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import os
import joblib
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for creating and transforming features
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.categorical_columns = []
        self.numerical_columns = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        logger.info("Creating new features...")
        df_feat = df.copy()
        
        # 1. Tenure groups
        df_feat['tenure_group'] = pd.cut(
            df_feat['tenure'],
            bins=[0, 12, 24, 48, 60, np.inf],
            labels=['0-1yr', '1-2yr', '2-4yr', '4-5yr', '5+yr']
        )
        
        # 2. Monthly charges groups
        df_feat['monthly_charges_group'] = pd.cut(
            df_feat['MonthlyCharges'],
            bins=[0, 30, 60, 90, np.inf],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # 3. Average monthly spend (TotalCharges / tenure)
        df_feat['avg_monthly_spend'] = np.where(
            df_feat['tenure'] > 0,
            df_feat['TotalCharges'] / df_feat['tenure'],
            df_feat['MonthlyCharges']
        )
        
        # 4. Has multiple services
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count services (excluding 'No' and 'No internet service')
        def count_services(row):
            count = 0
            for col in service_cols:
                if col in row.index:
                    if row[col] not in ['No', 'No internet service', 'No phone service']:
                        count += 1
            return count
        
        df_feat['num_services'] = df_feat.apply(count_services, axis=1)
        
        # 5. Has partner and dependents
        df_feat['has_family'] = np.where(
            (df_feat['Partner'] == 'Yes') | (df_feat['Dependents'] == 'Yes'),
            'Yes', 'No'
        )
        
        # 6. Contract type numeric (for risk scoring)
        contract_risk = {
            'Month-to-month': 3,
            'One year': 2,
            'Two year': 1
        }
        df_feat['contract_risk'] = df_feat['Contract'].map(contract_risk)
        
        # 7. Charges to tenure ratio
        df_feat['charges_per_tenure'] = df_feat['MonthlyCharges'] / (df_feat['tenure'] + 1)
        
        logger.info(f"Created {7} new features")
        
        return df_feat
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        categorical_cols: list,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical columns
            fit: Whether to fit encoders or use existing ones
            
        Returns:
            DataFrame with encoded categories
        """
        logger.info("Encoding categorical variables...")
        df_encoded = df.copy()
        
        self.categorical_columns = categorical_cols
        
        for col in categorical_cols:
            if col not in df_encoded.columns:
                continue
            
            if fit:
                # Fit new encoder
                le = LabelEncoder()
                # Handle unseen categories by converting to string first
                df_encoded[col] = df_encoded[col].astype(str)
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le
            else:
                # Use existing encoder
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df_encoded[col] = df_encoded[col].astype(str)
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        
        return df_encoded
    
    def scale_numerical(
        self, 
        df: pd.DataFrame, 
        numerical_cols: list,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical variables
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical columns
            fit: Whether to fit scaler or use existing one
            
        Returns:
            DataFrame with scaled numerical features
        """
        logger.info("Scaling numerical variables...")
        df_scaled = df.copy()
        
        self.numerical_columns = numerical_cols
        
        # Filter columns that exist
        existing_cols = [col for col in numerical_cols if col in df_scaled.columns]
        
        if fit:
            df_scaled[existing_cols] = self.scaler.fit_transform(df_scaled[existing_cols])
        else:
            df_scaled[existing_cols] = self.scaler.transform(df_scaled[existing_cols])
        
        logger.info(f"Scaled {len(existing_cols)} numerical columns")
        
        return df_scaled
    
    def save_artifacts(self, path: str):
        """
        Save feature engineering artifacts (encoders, scaler)
        
        Args:
            path: Directory to save artifacts
        """
        os.makedirs(path, exist_ok=True)
        
        # Save label encoders
        joblib.dump(self.label_encoders, os.path.join(path, 'label_encoders.joblib'))
        
        # Save scaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.joblib'))
        
        # Save column info
        column_info = {
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_names': self.feature_names
        }
        joblib.dump(column_info, os.path.join(path, 'column_info.joblib'))
        
        logger.info(f"Saved feature engineering artifacts to {path}")
    
    def load_artifacts(self, path: str):
        """
        Load feature engineering artifacts
        
        Args:
            path: Directory with saved artifacts
        """
        self.label_encoders = joblib.load(os.path.join(path, 'label_encoders.joblib'))
        self.scaler = joblib.load(os.path.join(path, 'scaler.joblib'))
        
        column_info = joblib.load(os.path.join(path, 'column_info.joblib'))
        self.categorical_columns = column_info['categorical_columns']
        self.numerical_columns = column_info['numerical_columns']
        self.feature_names = column_info['feature_names']
        
        logger.info(f"Loaded feature engineering artifacts from {path}")


def build_features(
    input_path: str,
    output_path: str,
    artifacts_path: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Main feature building function
    
    Args:
        input_path: Path to cleaned data
        output_path: Path to save featured data
        artifacts_path: Path to save feature artifacts
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Load cleaned data
    logger.info(f"Loading cleaned data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Separate target
    target = df['Churn'].copy()
    df = df.drop('Churn', axis=1)
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Create new features
    df = fe.create_features(df)
    
    # Define column types
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen',
        'tenure_group', 'monthly_charges_group', 'has_family'
    ]
    
    numerical_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'avg_monthly_spend', 'num_services', 'contract_risk',
        'charges_per_tenure'
    ]
    
    # Encode categorical variables
    df = fe.encode_categorical(df, categorical_cols, fit=True)
    
    # Scale numerical variables
    df = fe.scale_numerical(df, numerical_cols, fit=True)
    
    # Store feature names
    fe.feature_names = list(df.columns)
    
    # Save artifacts
    fe.save_artifacts(artifacts_path)
    
    # Save featured data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine features and target for saving
    df_final = df.copy()
    df_final['Churn'] = target
    df_final.to_csv(output_path, index=False)
    
    logger.info(f"Saved featured data to {output_path}")
    logger.info(f"Final feature set: {len(fe.feature_names)} features")
    
    return df, target


if __name__ == "__main__":
    # Run feature engineering
    input_path = "data/processed/cleaned_data.csv"
    output_path = "data/processed/featured_data.csv"
    artifacts_path = "models/feature_artifacts"
    
    X, y = build_features(input_path, output_path, artifacts_path)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nTarget distribution:\n{y.value_counts()}")