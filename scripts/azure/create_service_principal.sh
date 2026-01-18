#!/bin/bash

# ============================================
# Create Azure Service Principal for GitHub Actions
# ============================================

# Load credentials
source azure_credentials.env

SUBSCRIPTION_ID=$(az account show --query "id" -o tsv)
SP_NAME="sp-github-mlops-churn"

echo "Creating Service Principal: $SP_NAME"

# Create service principal with contributor access
SP_OUTPUT=$(az ad sp create-for-rbac \
    --name $SP_NAME \
    --role contributor \
    --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \
    --sdk-auth)

echo ""
echo "============================================"
echo "GITHUB SECRET: AZURE_CREDENTIALS"
echo "============================================"
echo "Copy the following JSON and add it as a GitHub secret named 'AZURE_CREDENTIALS':"
echo ""
echo "$SP_OUTPUT"
echo ""
echo "============================================"

# Save to file
echo "$SP_OUTPUT" > azure_sp_credentials.json
echo "Also saved to: azure_sp_credentials.json"
echo "DO NOT commit this file to git!"