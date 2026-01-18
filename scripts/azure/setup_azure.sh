#!/bin/bash

# ============================================
# Azure Resources Setup Script
# ============================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================
# Configuration - MODIFY THESE VALUES
# ============================================
RESOURCE_GROUP="rg-mlops-churn"
LOCATION="eastus"
ACR_NAME="acrmlops$(openssl rand -hex 4)"  # Must be globally unique
STORAGE_ACCOUNT="stmlops$(openssl rand -hex 4)"  # Must be globally unique
CONTAINER_APP_ENV="cae-mlops-churn"
CONTAINER_APP_NAME="ca-churn-api"
LOG_ANALYTICS_WORKSPACE="law-mlops-churn"
APP_INSIGHTS_NAME="appi-mlops-churn"

echo "============================================"
echo "Azure MLOps Setup"
echo "============================================"
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "ACR Name: $ACR_NAME"
echo "Storage Account: $STORAGE_ACCOUNT"
echo "============================================"

# ============================================
# 1. Create Resource Group
# ============================================
print_status "Creating Resource Group..."
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# ============================================
# 2. Create Azure Container Registry
# ============================================
print_status "Creating Azure Container Registry..."
az acr create \
    --resource-group $RESOURCE_GROUP \
    --name $ACR_NAME \
    --sku Basic \
    --admin-enabled true

# Get ACR credentials
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query "loginServer" -o tsv)

print_status "ACR Login Server: $ACR_LOGIN_SERVER"

# ============================================
# 3. Create Storage Account (for DVC)
# ============================================
print_status "Creating Storage Account for DVC..."
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --kind StorageV2

# Create container for DVC
az storage container create \
    --name dvc-storage \
    --account-name $STORAGE_ACCOUNT

# Create container for models
az storage container create \
    --name models \
    --account-name $STORAGE_ACCOUNT

# Get storage connection string
STORAGE_CONNECTION_STRING=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --query "connectionString" -o tsv)

# ============================================
# 4. Create Log Analytics Workspace
# ============================================
print_status "Creating Log Analytics Workspace..."
az monitor log-analytics workspace create \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --location $LOCATION

LOG_ANALYTICS_WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --query "customerId" -o tsv)

LOG_ANALYTICS_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group $RESOURCE_GROUP \
    --workspace-name $LOG_ANALYTICS_WORKSPACE \
    --query "primarySharedKey" -o tsv)

# ============================================
# 5. Create Application Insights
# ============================================
print_status "Creating Application Insights..."
az monitor app-insights component create \
    --app $APP_INSIGHTS_NAME \
    --location $LOCATION \
    --resource-group $RESOURCE_GROUP \
    --workspace $LOG_ANALYTICS_WORKSPACE

APP_INSIGHTS_KEY=$(az monitor app-insights component show \
    --app $APP_INSIGHTS_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "instrumentationKey" -o tsv)

APP_INSIGHTS_CONNECTION_STRING=$(az monitor app-insights component show \
    --app $APP_INSIGHTS_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "connectionString" -o tsv)

# ============================================
# 6. Create Container Apps Environment
# ============================================
print_status "Creating Container Apps Environment..."
az containerapp env create \
    --name $CONTAINER_APP_ENV \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --logs-workspace-id $LOG_ANALYTICS_WORKSPACE_ID \
    --logs-workspace-key $LOG_ANALYTICS_KEY

# ============================================
# 7. Output Summary
# ============================================
echo ""
echo "============================================"
echo "AZURE SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Resource Group:     $RESOURCE_GROUP"
echo "ACR Login Server:   $ACR_LOGIN_SERVER"
echo "ACR Username:       $ACR_USERNAME"
echo "Storage Account:    $STORAGE_ACCOUNT"
echo "Container App Env:  $CONTAINER_APP_ENV"
echo ""
echo "============================================"
echo "SAVE THESE FOR GITHUB SECRETS:"
echo "============================================"
echo ""
echo "ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER"
echo "ACR_USERNAME=$ACR_USERNAME"
echo "ACR_PASSWORD=$ACR_PASSWORD"
echo "STORAGE_CONNECTION_STRING=$STORAGE_CONNECTION_STRING"
echo "APP_INSIGHTS_CONNECTION_STRING=$APP_INSIGHTS_CONNECTION_STRING"
echo ""

# Save to file
cat > azure_credentials.env << EOF
# Azure Credentials - DO NOT COMMIT THIS FILE
RESOURCE_GROUP=$RESOURCE_GROUP
ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER
ACR_USERNAME=$ACR_USERNAME
ACR_PASSWORD=$ACR_PASSWORD
STORAGE_ACCOUNT=$STORAGE_ACCOUNT
STORAGE_CONNECTION_STRING=$STORAGE_CONNECTION_STRING
LOG_ANALYTICS_WORKSPACE_ID=$LOG_ANALYTICS_WORKSPACE_ID
APP_INSIGHTS_KEY=$APP_INSIGHTS_KEY
APP_INSIGHTS_CONNECTION_STRING=$APP_INSIGHTS_CONNECTION_STRING
CONTAINER_APP_ENV=$CONTAINER_APP_ENV
CONTAINER_APP_NAME=$CONTAINER_APP_NAME
EOF

print_status "Credentials saved to azure_credentials.env"
print_warning "DO NOT commit azure_credentials.env to git!"