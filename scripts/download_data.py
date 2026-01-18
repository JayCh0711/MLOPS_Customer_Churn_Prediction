"""
Script to download the Telco Customer Churn dataset
"""
import os
import pandas as pd
import urllib.request

def download_dataset():
    """Download the Telco Customer Churn dataset"""
    
    # Create data directory if not exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Dataset URL (IBM Sample Dataset - publicly available)
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    output_path = "data/raw/telco_churn.csv"
    
    print("Downloading dataset...")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Dataset downloaded successfully to {output_path}")
        
        # Verify the download
        df = pd.read_csv(output_path)
        print(f"\nDataset Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()