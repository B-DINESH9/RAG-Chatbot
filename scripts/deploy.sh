#!/bin/bash
# Deploy RAG Chatbot to Google Cloud Run
#
# Prerequisites:
#   1. Google Cloud SDK (gcloud) installed
#   2. Docker installed
#   3. Authenticated: gcloud auth login
#
# Usage: ./scripts/deploy.sh [PROJECT_ID] [REGION] [SERVICE_NAME]

set -euo pipefail

PROJECT_ID=${1:-"your-gcp-project-id"}
REGION=${2:-"us-central1"}
SERVICE_NAME=${3:-"rag-chatbot"}
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${SERVICE_NAME}/${SERVICE_NAME}"

echo "======================================="
echo "  RAG Chatbot - Cloud Run Deployment"
echo "======================================="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE_NAME}"
echo "Image:    ${IMAGE_NAME}"
echo "======================================="

# Step 1: Set project
echo "\n[1/5] Setting GCP project..."
gcloud config set project ${PROJECT_ID}

# Step 2: Enable required APIs
echo "\n[2/5] Enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com

# Step 3: Create Artifact Registry repository (if not exists)
echo "\n[3/5] Creating Artifact Registry repository..."
gcloud artifacts repositories create ${SERVICE_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="RAG Chatbot Docker images" \
    2>/dev/null || echo "Repository already exists."

# Step 4: Build and push Docker image
echo "\n[4/5] Building and pushing Docker image..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
docker build -t ${IMAGE_NAME}:latest .
docker push ${IMAGE_NAME}:latest

# Step 5: Deploy to Cloud Run
echo "\n[5/5] Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image=${IMAGE_NAME}:latest \
    --region=${REGION} \
    --port=8501 \
    --memory=2Gi \
    --cpu=2 \
    --min-instances=0 \
    --max-instances=3 \
    --allow-unauthenticated \
    --set-env-vars="STREAMLIT_SERVER_HEADLESS=true"

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)')

echo "\n======================================="
echo "  Deployment Complete!"
echo "  URL: ${SERVICE_URL}"
echo "======================================="
