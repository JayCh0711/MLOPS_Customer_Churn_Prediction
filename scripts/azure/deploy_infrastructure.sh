#!/bin/bash

# ============================================
# Deploy Azure Infrastructure with Bicep
# ============================================

RESOURCE_GROUP="rg-mlops-churn"
LOCATION="eastus"
ENVIRONMENT=${1:-"dev"}

echo "Deploying infrastructure for environment: $ENVIRONMENT"

# Create resource group
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# Deploy Bicep template
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file infrastructure/main.bicep \
    --parameters environment=$ENVIRONMENT \
    --parameters location=$LOCATION

# Get outputs
echo ""
echo "============================================"
echo "Deployment Outputs:"
echo "============================================"
az deployment group show \
    --resource-group $RESOURCE_GROUP \
    --name main \
    --query "properties.outputs" -o json