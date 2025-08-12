#!/bin/bash

# OpenTuneWeaver RunPod Deployment Script

echo "🏗️ Building OpenTuneWeaver Docker Image..."

# Build Docker Image
docker build -t opentuneweaver:latest .

# Tag für Docker Hub (ersetze USERNAME mit deinem Docker Hub Username)
docker tag opentuneweaver:latest USERNAME/opentuneweaver:latest

# Push to Docker Hub
echo "📤 Pushing to Docker Hub..."
docker push USERNAME/opentuneweaver:latest

echo "✅ Image ready for RunPod deployment!"
echo ""
echo "📋 RunPod Configuration:"
echo "  - Docker Image: USERNAME/opentuneweaver:latest"
echo "  - Container Disk: 50 GB"
echo "  - Volume Disk: 100 GB"
echo "  - Volume Mount Path: /workspace"
echo "  - Exposed Ports: 8080"
echo "  - Environment Variables:"
echo "    HF_TOKEN=your_huggingface_token"
echo ""