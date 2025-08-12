#!/bin/bash

# OpenTuneWeaver All-in-One Build Script

echo "🏗️ Building OpenTuneWeaver with integrated Ollama..."

# Build Docker Image
docker build -f Dockerfile -t opentuneweaver-aio:latest .

# Tag für Docker Hub
DOCKER_USERNAME=${DOCKER_USERNAME:-"yourusername"}
docker tag opentuneweaver-aio:latest $DOCKER_USERNAME/opentuneweaver-aio:latest

# Optional: Test lokal
read -p "Test locally first? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    docker-compose -f docker-compose-ollama.yml up -d
    echo "🧪 Testing locally..."
    sleep 30
    docker exec otw-aio-container python /workspace/OpenTuneWeaver/scripts/health_check.py
    docker-compose -f docker-compose-ollama.yml logs --tail=50
    read -p "Continue with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f docker-compose-ollama.yml down
        exit 0
    fi
    docker-compose -f docker-compose-ollama.yml down
fi

# Push to Docker Hub
echo "📤 Pushing to Docker Hub..."
docker push $DOCKER_USERNAME/opentuneweaver-aio:latest

echo "✅ Image ready for RunPod!"
echo ""
echo "📋 RunPod Configuration:"
echo "  - Docker Image: $DOCKER_USERNAME/opentuneweaver-aio:latest"
echo "  - Container Disk: 100 GB (für Ollama Models)"
echo "  - Volume Disk: 100 GB"
echo "  - Volume Mount Path: /workspace"
echo "  - Exposed Ports: 8080,11434"
echo "  - Min GPU: RTX 3090 24GB / A5000 24GB"
echo "  - Recommended: A6000 48GB / A100 40GB"
echo ""
echo "⚠️ Note: First start will take ~10-15 minutes to download gemma3:12b"