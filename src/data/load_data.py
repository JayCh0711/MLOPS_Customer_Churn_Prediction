"""
Data loading utilities for the MLOps project
"""
import pandas as pd
import os
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str = "data/raw/telco_churn.csv") -> pd.DataFrame:
    """
    Load raw data from CSV file
    
    Args:
        filepath: Path to the raw data file
        
    Returns:
        DataFrame with raw data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    return df


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with data information
    """
    info = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    return info


def validate_data(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate the loaded data
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check if DataFrame is empty
    if len(df) == 0:
        issues.append("DataFrame is empty")
    
    # Check for required columns
    required_columns = ['customerID', 'Churn']
    for col in required_columns:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")
    
    # Check for duplicate customer IDs
    if 'customerID' in df.columns:
        duplicates = df['customerID'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate customer IDs")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("Data validation passed")
    else:
        logger.warning(f"Data validation failed with {len(issues)} issues")
    
    return is_valid, issues


if __name__ == "__main__":
    # Test the data loading
    df = load_raw_data()
    info = get_data_info(df)
    print("\n=== Data Information ===")
    print(f"Rows: {info['n_rows']}")
    print(f"Columns: {info['n_columns']}")
    print(f"Memory: {info['memory_usage_mb']:.2f} MB")
    
    is_valid, issues = validate_data(df)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")