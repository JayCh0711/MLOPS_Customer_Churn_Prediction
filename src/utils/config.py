"""
Configuration loader utility
"""
import yaml
import os
from typing import Any, Dict

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_config() -> Dict[str, Any]:
    """Get data configuration section"""
    config = load_config()
    return config.get('data', {})


def get_feature_config() -> Dict[str, Any]:
    """Get feature configuration section"""
    config = load_config()
    return config.get('features', {})


def get_model_config() -> Dict[str, Any]:
    """Get model configuration section"""
    config = load_config()
    return config.get('model', {})


def get_mlflow_config() -> Dict[str, Any]:
    """Get MLflow configuration section"""
    config = load_config()
    return config.get('mlflow', {})


if __name__ == "__main__":
    # Test config loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"\nData config: {get_data_config()}")
    print(f"\nModel config: {get_model_config()}")