#!/bin/bash

# ============================================
# Deploy to Azure Container Apps
# ============================================

set -e

# Load credentials
source azure_credentials.env

IMAGE_TAG=${1:-"latest"}
FULL_IMAGE_NAME="$ACR_LOGIN_SERVER/churn-prediction-api:$IMAGE_TAG"

echo "============================================"
echo "Deploying to Azure Container Apps"
echo "============================================"
echo "Image: $FULL_IMAGE_NAME"
echo "============================================"

# Login to ACR
echo "Logging into Azure Container Registry..."
az acr login --name ${ACR_LOGIN_SERVER%%.*}

# Build and push image
echo "Building and pushing Docker image..."
docker build -t $FULL_IMAGE_NAME .
docker push $FULL_IMAGE_NAME

# Check if Container App exists
APP_EXISTS=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "name" -o tsv 2>/dev/null || echo "")

if [ -z "$APP_EXISTS" ]; then
    echo "Creating new Container App..."
    az containerapp create \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --environment $CONTAINER_APP_ENV \
        --image $FULL_IMAGE_NAME \
        --registry-server $ACR_LOGIN_SERVER \
        --registry-username $ACR_USERNAME \
        --registry-password $ACR_PASSWORD \
        --target-port 8000 \
        --ingress external \
        --min-replicas 1 \
        --max-replicas 10 \
        --cpu 0.5 \
        --memory 1.0Gi \
        --env-vars \
            ENVIRONMENT=production \
            APP_INSIGHTS_CONNECTION_STRING="$APP_INSIGHTS_CONNECTION_STRING" \
        --query "properties.configuration.ingress.fqdn" -o tsv
else
    echo "Updating existing Container App..."
    az containerapp update \
        --name $CONTAINER_APP_NAME \
        --resource-group $RESOURCE_GROUP \
        --image $FULL_IMAGE_NAME \
        --query "properties.configuration.ingress.fqdn" -o tsv
fi

# Get the app URL
APP_URL=$(az containerapp show \
    --name $CONTAINER_APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" -o tsv)

echo ""
echo "============================================"
echo "DEPLOYMENT COMPLETE!"
echo "============================================"
echo "API URL: https://$APP_URL"
echo "Health: https://$APP_URL/health"
echo "Docs:   https://$APP_URL/docs"
echo "============================================"

# Test the deployment
echo ""
echo "Testing deployment..."
sleep 10
curl -s "https://$APP_URL/health" | python -m json.tool