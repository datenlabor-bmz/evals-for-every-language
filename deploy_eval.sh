#!/bin/bash

echo "Deploying AI Language Evaluation to Google Cloud Run"
echo "Cost limit: $20 USD"
echo "No runtime limit - will run to completion"

# Build the Docker image first
echo "🔨 Building Docker image..."
gcloud builds submit --config cloudbuild.yaml .

# Deploy the built image
echo "🚀 Deploying to Cloud Run..."
gcloud run deploy ai-language-eval \
  --image gcr.io/ai-language-eval-1754052060/ai-language-eval \
  --region us-central1 \
  --platform managed \
  --memory 2Gi \
  --cpu 1 \
  --max-instances 1 \
  --timeout 3600 \
  --concurrency 1 \
  --no-allow-unauthenticated \
  --set-env-vars="N_SENTENCES=20,MAX_LANGUAGES=150,COST_LIMIT_USD=20" \
  --quiet

echo "✅ Deployment completed!"
echo "🔗 Service URL: $(gcloud run services describe ai-language-eval --region=us-central1 --format='value(status.url)')"
echo "📊 Monitor costs: https://console.cloud.google.com/billing/linkedaccount?project=ai-language-eval-1754052060"
echo "💾 Results will be saved to: gs://ai-language-eval-results/" 