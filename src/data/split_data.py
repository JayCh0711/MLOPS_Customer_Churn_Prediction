"""
Data splitting module for train/test split
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def split_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> dict:
    """
    Split data into train and test sets
    
    Args:
        input_path: Path to featured data
        output_dir: Directory to save split data
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by target
        
    Returns:
        Dictionary with split information
    """
    logger.info(f"Loading featured data from {input_path}")
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Features: {X.shape[1]}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    # Perform split
    stratify_col = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    logger.info(f"Train set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    logger.info(f"Saved split data to {output_dir}")
    
    # Return split info
    split_info = {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'train_target_distribution': y_train.value_counts().to_dict(),
        'test_target_distribution': y_test.value_counts().to_dict(),
        'test_size': test_size,
        'random_state': random_state
    }
    
    return split_info


def load_split_data(data_dir: str = "data/processed") -> tuple:
    """
    Load split data from directory
    
    Args:
        data_dir: Directory containing split data
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).squeeze()
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load config
    config = load_config()
    
    # Run split
    input_path = "data/processed/featured_data.csv"
    output_dir = "data/processed"
    
    split_info = split_data(
        input_path=input_path,
        output_dir=output_dir,
        test_size=config['model']['test_size'],
        random_state=config['model']['random_state']
    )
    
    print("\n=== Split Information ===")
    for key, value in split_info.items():
        print(f"{key}: {value}")