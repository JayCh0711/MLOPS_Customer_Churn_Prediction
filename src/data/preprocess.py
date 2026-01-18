"""
Data preprocessing module for Telco Customer Churn dataset
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Data preprocessing class for cleaning and preparing raw data
    """
    
    def __init__(self):
        self.preprocessing_stats = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df_clean = df.copy()
        
        # Store original shape
        original_shape = df_clean.shape
        
        # 1. Handle TotalCharges - convert to numeric
        # TotalCharges has some empty strings that need to be handled
        df_clean['TotalCharges'] = pd.to_numeric(
            df_clean['TotalCharges'], 
            errors='coerce'
        )
        
        # 2. Fill missing TotalCharges with 0 (new customers)
        missing_total_charges = df_clean['TotalCharges'].isnull().sum()
        df_clean['TotalCharges'].fillna(0, inplace=True)
        logger.info(f"Filled {missing_total_charges} missing TotalCharges values")
        
        # 3. Convert SeniorCitizen to string for consistency
        df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        # 4. Convert Churn to binary
        df_clean['Churn'] = df_clean['Churn'].map({'No': 0, 'Yes': 1})
        
        # 5. Remove duplicates
        duplicates = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {duplicates} duplicate rows")
        
        # 6. Remove customerID (not a feature)
        if 'customerID' in df_clean.columns:
            self.customer_ids = df_clean['customerID'].copy()
            df_clean = df_clean.drop('customerID', axis=1)
            logger.info("Removed customerID column")
        
        # Store preprocessing stats
        self.preprocessing_stats = {
            'original_shape': original_shape,
            'cleaned_shape': df_clean.shape,
            'missing_total_charges_filled': missing_total_charges,
            'duplicates_removed': duplicates
        }
        
        logger.info(f"Data cleaning complete. Shape: {df_clean.shape}")
        
        return df_clean
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        columns: list,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns
            method: Method to handle outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers using {method} method...")
        df_out = df.copy()
        
        for col in columns:
            if col not in df_out.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_out[col].quantile(0.25)
                Q3 = df_out[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing
                outliers_count = ((df_out[col] < lower_bound) | (df_out[col] > upper_bound)).sum()
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
                
                logger.info(f"  {col}: capped {outliers_count} outliers")
                
            elif method == 'zscore':
                mean = df_out[col].mean()
                std = df_out[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df_out
    
    def get_stats(self) -> dict:
        """Return preprocessing statistics"""
        return self.preprocessing_stats


def preprocess_data(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Main preprocessing function
    
    Args:
        input_path: Path to raw data
        output_path: Path to save processed data
        
    Returns:
        Processed DataFrame
    """
    # Load raw data
    logger.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Clean data
    df_clean = preprocessor.clean_data(df)
    
    # Handle outliers in numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_clean = preprocessor.handle_outliers(df_clean, numerical_cols)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    # Print stats
    stats = preprocessor.get_stats()
    logger.info(f"Preprocessing stats: {stats}")
    
    return df_clean


if __name__ == "__main__":
    # Run preprocessing
    input_path = "data/raw/telco_churn.csv"
    output_path = "data/processed/cleaned_data.csv"
    
    df = preprocess_data(input_path, output_path)
    print(f"\nProcessed data shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nTarget distribution:\n{df['Churn'].value_counts()}")