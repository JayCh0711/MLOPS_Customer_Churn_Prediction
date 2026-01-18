#!/bin/bash

# ============================================
# Configure DVC with Azure Blob Storage
# ============================================

# Load Azure credentials
source azure_credentials.env

echo "Configuring DVC with Azure Blob Storage..."

# Install Azure DVC extension
pip install dvc-azure

# Remove existing remote (if any)
dvc remote remove azure 2>/dev/null || true

# Add Azure remote
dvc remote add -d azure azure://$STORAGE_ACCOUNT/dvc-storage

# Configure authentication
dvc remote modify azure connection_string "$STORAGE_CONNECTION_STRING"

# Verify configuration
echo ""
echo "DVC Configuration:"
cat .dvc/config

echo ""
echo "DVC Azure remote configured successfully!"
echo "Run 'dvc push' to upload data to Azure Blob Storage"